"""Verify that publishing a note preserves other contributors and requires read-back."""
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'skills/paper-ingest/scripts/sync_to_lab.py'
spec = importlib.util.spec_from_file_location('library_client', PATH)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

class LibraryClientTest(unittest.TestCase):
    def setUp(self):
        self.paper = {'id':'paper-id','title':'A paper','arxiv_id':'fixture','obsidian_path':'Literature/Topic/Paper'}
        self.chunks = [{'section':'Review','content':'Detailed old reading'},
                       {'section':'Original abstract','content':'Original source passage'}]
        self.incoming = {**{k:v for k,v in self.paper.items() if k!='id'},
                         'sections':[{'section':'Review','content':'Updated reading'}]}
        self.calls=[]
        self.claims=[]

    def rpc(self, name, args):
        self.calls.append((name,copy.deepcopy(args)))
        if name=='get_paper':
            return {'paper':copy.deepcopy(self.paper),'chunks':copy.deepcopy(self.chunks),
                    'claims':copy.deepcopy(self.claims)} if self.paper else None
        if name=='upsert_paper':
            self.assertNotIn('sections',args)
            self.assertNotIn('claims',args)
            if self.paper is None:self.paper={'id':'new-paper-id'}
            self.paper.update(args)
            if "themes" in args:self.paper["theme_slugs"]=args["themes"]
        if name=='record_paper_claim':
            saved={k:v for k,v in args.items() if k!='paper_id'}
            if saved not in self.claims:self.claims.append(saved)
        if name=='add_paper_section':
            self.chunks=[x for x in self.chunks if x['section']!=args['section']]+[dict(args)]
        return {}

    def test_claims_are_published_and_verified_for_new_and_existing_papers(self):
        for new in (False, True):
            with self.subTest(new=new):
                self.setUp()
                if new:self.paper=None;self.chunks=[]
                self.incoming['claims']=[{'statement':'A bounded result','locator':'Section 3','kind':'theoretical'}]
                with patch.object(client,'_ask',side_effect=self.rpc):
                    client.sync_paper(self.incoming)
                    client.sync_paper(self.incoming)
                self.assertEqual(len(self.claims),1)
                self.assertEqual(self.claims[0]['statement'],'A bounded result')
                self.assertEqual(self.claims[0]['quote'],'')
                self.assertEqual(self.calls[-1][0],'get_paper')

    def test_dropped_claim_is_not_reported_as_verified(self):
        self.incoming['claims']=[{'statement':'A result'}]
        def rpc(name,args):
            return {} if name=='record_paper_claim' else self.rpc(name,args)
        with patch.object(client,'_ask',side_effect=rpc):
            with self.assertRaisesRegex(RuntimeError,'Read-back differs'):
                client.sync_paper(self.incoming)

    def test_concurrently_created_paper_preserves_its_sections(self):
        self.incoming['obsidian_path']='Literature/Other/Paper'
        first=True
        def rpc(name,args):
            nonlocal first
            if first and name=='get_paper':
                first=False
                return None
            return self.rpc(name,args)
        with patch.object(client,'_ask',side_effect=rpc):
            client.sync_paper(self.incoming)
        self.assertEqual(self.paper['obsidian_path'],'Literature/Topic/Paper')
        self.assertIn('Detailed old reading',[x['content'] for x in self.chunks])
        self.assertIn('Updated reading',[x['content'] for x in self.chunks])

    def test_update_preserves_original_and_concurrent_extra_sections(self):
        def rpc(name,args):
            if name=='add_paper_section':
                self.chunks.append({'section':'Another contributor','content':'Arrived after the read'})
            return self.rpc(name,args)
        with patch.object(client,'_ask',side_effect=rpc):
            client.sync_paper(self.incoming)
        self.assertEqual({x['section'] for x in self.chunks},
                         {'Review','Original abstract','Another contributor'})
        self.assertEqual(self.calls[-1][0],'get_paper')

    def test_other_note_preserves_both_readings_on_retry(self):
        self.incoming['obsidian_path']='Literature/Other/Paper'
        with patch.object(client,'_ask',side_effect=self.rpc):
            client.sync_paper(self.incoming)
            client.sync_paper(self.incoming)
        sections = {x['section']: x['content'] for x in self.chunks}
        self.assertEqual(sections['Review'], 'Detailed old reading')
        self.assertEqual(sections['Original abstract'], 'Original source passage')
        self.assertIn('Updated reading', sections.values())
        self.assertEqual(len(sections), 3)
        self.assertEqual(self.paper['obsidian_path'], 'Literature/Topic/Paper')
        self.assertEqual(self.calls[-1][0], 'get_paper')

    def test_same_basename_from_another_folder_keeps_separate_readings(self):
        with patch.object(client,'_ask',side_effect=self.rpc):
            for folder, content in [('Other', 'Second reading'), ('Third', 'Third reading')]:
                self.incoming['obsidian_path'] = f'Literature/{folder}/Paper'
                self.incoming['sections'][0]['content'] = content
                client.sync_paper(self.incoming)
        self.assertEqual({x['content'] for x in self.chunks},
                         {'Detailed old reading', 'Original source passage',
                          'Second reading', 'Third reading'})

    def test_silent_noop_is_not_reported_as_success(self):
        with patch.object(client,'_ask',side_effect=lambda name,args:
                          self.rpc(name,args) if name=='get_paper' else {}):
            with self.assertRaisesRegex(RuntimeError,'Read-back differs'):
                client.sync_paper(self.incoming)

    def test_read_failure_cannot_be_treated_as_new_paper(self):
        with patch.object(client,'_ask',side_effect=RuntimeError('unreachable')) as rpc:
            with self.assertRaisesRegex(RuntimeError,'unreachable'):
                client.sync_paper(self.incoming)
            self.assertEqual(rpc.call_count,1)

    def test_themes_and_tags_from_other_contributors_survive(self):
        self.paper.update(theme_slugs=['other-theme'],tags=['curated'])
        self.incoming.update(tags=['from-note'],summary_ru='Updated summary')
        with patch.object(client,'_ask',side_effect=self.rpc):client.sync_paper(self.incoming)
        self.assertEqual(self.paper['theme_slugs'],['other-theme'])
        self.assertEqual(self.paper['tags'],['curated','from-note'])

    def test_summary_write_must_be_verified_too(self):
        self.incoming['summary_ru']='New summary'
        def rpc(name,args):
            answer=self.rpc(name,args)
            if name=='upsert_paper':self.paper['summary_ru']='Old summary'
            return answer
        with patch.object(client,'_ask',side_effect=rpc):
            with self.assertRaisesRegex(RuntimeError,'Read-back differs'):client.sync_paper(self.incoming)

    def test_empty_metadata_does_not_erase_existing_authors(self):
        self.paper['authors']=['Original author']
        self.incoming['authors']=[]
        with patch.object(client,'_ask',side_effect=self.rpc):client.sync_paper(self.incoming)
        self.assertEqual(self.paper['authors'],['Original author'])

if __name__=='__main__':unittest.main()
