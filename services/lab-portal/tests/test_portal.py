"""Раздача ключей: одна ссылка, одна кнопка, один ключ. Паролей здесь нет."""

import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portal  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    return portal.Store(tmp_path / "portal.db")


def test_an_invite_works_once(store) -> None:
    """Ссылка ходит по переписке, поэтому второй раз она не должна открывать ничего."""
    token = store.invite("student@lab")

    assert store.invited_email(token) == "student@lab"
    assert store.claim(token) == "student@lab"
    assert store.claim(token) is None, "второй раз погасить нечего"
    assert store.invited_email(token) is None
    assert store.invited_email("выдуманная ссылка") is None


def test_only_one_of_many_presses_gets_the_key(store) -> None:
    """Две вкладки, нажатые разом, проходили проверку обе — и одна ссылка давала пять ключей."""
    import concurrent.futures

    token = store.invite("student@lab")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.claim(token), range(8)))

    assert results.count("student@lab") == 1, results
    assert results.count(None) == 7


def test_the_store_survives_many_threads(store) -> None:
    """Одно соединение SQLite на все потоки ломалось на обычном двойном нажатии кнопки."""
    import concurrent.futures

    def work(number: int) -> str:
        token = store.invite(f"s{number}@lab")
        store.invited_email(token)
        return store.claim(token) or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        done = list(pool.map(work, range(120)))

    assert all(done), "ни один запрос не должен упасть на общем соединении"


def test_an_invite_left_lying_in_the_mail_expires(store, monkeypatch) -> None:
    """Ссылка путешествует открытым текстом через почту; вечной она быть не должна."""
    token = store.invite("student@lab")
    monkeypatch.setattr(portal, "now", lambda: portal.time.time().__int__() + store.LIFETIME + 60)

    assert store.invited_email(token) is None
    assert store.claim(token) is None


def test_a_base_that_answers_wrong_does_not_burn_the_invite(running) -> None:
    """Ответ без ключа гасил приглашение и оставлял человека с пустой страницей."""
    url, store, base = running
    token = store.invite("student@lab")

    def wrong(tool, arguments):
        if tool == "list_agent_tokens":
            return {"items": []}
        return {"вместо ключа": "ничего"}

    base.call = wrong
    answer = urllib.request.urlopen(f"{url}/invite/{token}", data=b"").read().decode()

    assert "Ключ не выдан" in answer
    assert store.invited_email(token) == "student@lab", "приглашение осталось годным"


class _FakeBase:
    def __init__(self):
        self.issued = []
        self.revoked = []
        self.live = []
        self.broken = False

    def call(self, tool, arguments):
        if self.broken:
            raise RuntimeError("база молчит")
        if tool == "list_agent_tokens":
            return {"items": [{"label": label, "created_at": "2026-09-01"} for label in self.live]}
        if tool == "issue_agent_token":
            self.issued.append(arguments["label"])
            self.live.append(arguments["label"])
            return {"token": "выданный-ключ", "label": arguments["label"]}
        if tool == "revoke_agent_token":
            count = self.live.count(arguments["label"])
            self.live = [item for item in self.live if item != arguments["label"]]
            self.revoked.append(arguments["label"])
            return {"revoked": count}
        raise AssertionError(tool)


@pytest.fixture()
def running(tmp_path):
    portal.Portal.store = portal.Store(tmp_path / "portal.db")
    portal.Portal.base = _FakeBase()
    portal.Portal.public_url = "https://example.org:9443/mcp"
    server = ThreadingHTTPServer(("127.0.0.1", 0), portal.Portal)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", portal.Portal.store, portal.Portal.base
    server.shutdown()


def test_a_student_opens_the_link_and_gets_a_key(running) -> None:
    """Весь путь: ссылка, кнопка, ключ с готовой настройкой. Ни пароля, ни входа."""
    url, store, base = running
    token = store.invite("student@lab")

    offer = urllib.request.urlopen(f"{url}/invite/{token}").read().decode()
    assert "student@lab" in offer
    assert "Показать ключ" in offer
    assert "выданный-ключ" not in offer, "открытие ссылки ключа ещё не выдаёт"

    given = urllib.request.urlopen(f"{url}/invite/{token}", data=b"").read().decode()
    assert "выданный-ключ" in given
    assert "https://example.org:9443/mcp" in given
    assert "bearer-token-env-var" in given, "у Codex ключ берётся из переменной окружения"
    assert "claude mcp add" in given, "для Claude Code даётся его собственная команда"
    assert len(base.issued) == 1


def test_opening_the_link_does_not_burn_it(running) -> None:
    """Почтовые клиенты ходят по ссылкам сами ради предпросмотра: приглашение обязано выжить."""
    url, store, base = running
    token = store.invite("student@lab")

    urllib.request.urlopen(f"{url}/invite/{token}").read()
    urllib.request.urlopen(f"{url}/invite/{token}").read()

    assert store.invited_email(token) == "student@lab", "ссылка ещё жива"
    assert base.issued == [], "и ничего не выдано"


def test_a_used_link_gives_nothing(running) -> None:
    """Второе нажатие не должно давать второй ключ: ссылка одноразовая."""
    url, store, base = running
    token = store.invite("student@lab")

    urllib.request.urlopen(f"{url}/invite/{token}", data=b"").read()
    again = urllib.request.urlopen(f"{url}/invite/{token}").read().decode()

    assert "уже использована" in again
    assert len(base.issued) == 1


def test_a_new_key_does_not_break_the_old_one(running) -> None:
    """Человек мог уже настроить ноутбук и сервер: вторая ссылка не должна ломать то, что работает."""
    url, store, base = running
    urllib.request.urlopen(f"{url}/invite/{store.invite('student@lab')}", data=b"").read()
    second = urllib.request.urlopen(f"{url}/invite/{store.invite('student@lab')}", data=b"").read().decode()

    assert base.revoked == [], "ничего не погашено"
    assert len(base.live) == 2, "оба ключа живы"
    assert len(set(base.issued)) == 2, "имена разные, иначе их не различить в списке"
    assert "продолжают работать" in second, "человеку сказано, что прежние ключи целы"


def test_keys_issued_on_the_same_day_are_still_told_apart(running) -> None:
    """Гася потерянный ключ, ведущий должен понимать, какой из трёх он гасит."""
    url, store, base = running
    for _ in range(3):
        urllib.request.urlopen(f"{url}/invite/{store.invite('student@lab')}", data=b"").read()

    assert len(set(base.issued)) == 3, base.issued
    assert all(label.startswith(base.issued[0][:10]) for label in base.issued), "имя по дате выдачи"


def test_a_silent_base_does_not_burn_the_invite(running) -> None:
    """Человек не виноват, что база не ответила, и остаться без доступа он не должен."""
    url, store, base = running
    token = store.invite("student@lab")
    base.broken = True

    answer = urllib.request.urlopen(f"{url}/invite/{token}", data=b"").read().decode()

    assert "Ключ не выдан" in answer
    assert store.invited_email(token) == "student@lab", "ссылка осталась годной"


def test_a_stranger_is_told_where_to_get_a_link(running) -> None:
    url, _store, _base = running

    answer = urllib.request.urlopen(url + "/").read().decode()

    assert "попросите её у ведущего" in answer.lower()


def test_a_link_with_a_trailing_slash_still_works(running) -> None:
    """Почтовые клиенты дописывают косую: годная ссылка не должна выглядеть использованной."""
    url, store, _base = running
    token = store.invite("student@lab")

    for tail in ("/", ".", ")"):
        answer = urllib.request.urlopen(f"{url}/invite/{token}{tail}").read().decode()
        assert "Показать ключ" in answer, tail


def test_the_token_never_reaches_the_log(running, capfd) -> None:
    """Журнал лежит на публичной машине: живой ссылке в нём не место ни в каком написании."""
    url, store, _base = running
    token = store.invite("student@lab")

    for path in (f"/invite/{token}", f"/INVITE/{token}", f"//invite/{token}/x"):
        with suppress_error():
            urllib.request.urlopen(url + path).read()

    printed = capfd.readouterr().out
    assert token not in printed, printed[:200]


def suppress_error():
    import contextlib
    import urllib.error
    return contextlib.suppress(urllib.error.HTTPError, urllib.error.URLError)


def test_the_key_page_forbids_caching(running) -> None:
    """Страница показывает ключ: ни браузеру, ни посреднику держать её у себя нельзя."""
    url, store, _base = running
    token = store.invite("student@lab")

    answer = urllib.request.urlopen(f"{url}/invite/{token}", data=b"")

    assert "no-store" in answer.headers.get("Cache-Control", "")
    assert "frame-ancestors" in answer.headers.get("Content-Security-Policy", "")
