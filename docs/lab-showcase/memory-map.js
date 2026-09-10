/* Explanatory maps contain no private records and never call the live MCP. */
window.LAB_MEMORY_MAP = (() => {
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const origins = {
    projects:['Проект','create-project · контекст'],
    literature:['Литература','Hermes · paper-ingest'],
    ideation:['Идеи','research-ideation'],
    calls:['Созвоны','Brain Call · call-notes'],
    experiments:['Эксперименты','Hermes · coding agents'],
    results:['Результаты','анализ · academic-plotting'],
    writing:['Статья и ревью','ML Paper Writing · Astar'],
    presentation:['Публикация','paper-to-social · presentation']
  };
  const dot = '<i class="mem-star" aria-hidden="true"></i>';
  const arrow = '<i class="mem-inline-arrow" aria-hidden="true">→</i>';
  const station = (id, label, title, body, extra='') => `<article class="mem-station ${extra}" data-point="${id}">${dot}<span class="mem-label">${label}</span><h3>${title}</h3>${body}</article>`;
  const api = (...names) => `<span class="mem-api-names">${names.map(n=>`<code>${n}</code>`).join(' · ')}</span>`;
  const svg = '<svg class="mem-connections" aria-hidden="true"></svg>';
  const inlets = {
    ingest:[['scout','Литературный Hermes','находит по теме'],['search','paper-search','предлагает кандидатов'],['alphaxiv','alphaXiv / arXiv','ссылка на работу'],['social','X / коллеги','делятся статьёй'],['reader','Исследователь','выбирает для разбора']],
    project:[['hermes','Экспериментальный Hermes','прогоны и результаты'],['agents','Codex / Claude Code','код, проверки, выводы'],['call','Brain Call / call-notes','идеи и решения встречи'],['person','Исследователь','гипотезы и материалы']]
  };
  const feeders = kind => `<div class="mem-feeders" aria-label="${kind==='ingest'?'Откуда приходит статья':'Откуда приходят материалы проекта'}">${inlets[kind].map(([id,title,caption])=>`<div class="mem-inlet" data-inlet="${kind}" data-point="${kind}-via-${id}"><b>${title}</b><small>${caption}</small></div>`).join('')}</div>`;

  function overview(processes){
    return `<div class="mem-overview mem-diagram" data-map="overview">${svg}
      <div class="mem-origins" data-point="origins"><span class="mem-label">Вся работа в Atlas</span><div class="mem-source-stars">${processes.map(p=>`<div class="mem-origin" data-origin="${esc(p.id)}" style="--star:${esc(p.color)}"><i aria-hidden="true"></i><span><b>${origins[p.id][0]}</b><small>${origins[p.id][1]}</small></span></div>`).join('')}</div></div>
      <div class="mem-writer" data-point="writer">${dot}<div><b>Агент выделяет смысл</b><p>Задание / навык <span class="mem-trigger-arrow">↘</span> <span class="mem-hook-trigger">Hook · checkpoint ↗</span><br>Читает контекст, сохраняет важное и проверяет запись.</p></div></div>
      <div class="mem-main-lanes">
        <div class="mem-lane mem-literature-lane">
          ${station('paper','Из литературы','Статья','<p>Источник, полный текст<br>и разбор прочитанного.</p>')}
          ${station('extract','Разбор','Отдельные мысли','<p>Разделы → результаты, теоремы, методы и определения.</p>')}
        </div>
        <div class="mem-lane mem-research-lane">
          ${station('question','Наша работа','Проект','<p>Утверждения, протоколы<br>и результаты исследования.</p>','mem-project-node')}
          ${station('check','Проверка','Основания и оценка','<p>Эксперимент → измерения<br>или математический вывод.</p><small>Агент уточняет статус того же утверждения.</small>')}
        </div>
      </div>
      <article class="mem-assertion" data-point="assertion" aria-label="Что сохраняется в MCP: утверждение">
        <header><span class="mem-label">Lab Knowledge MCP · PostgreSQL</span><h2>Утверждение</h2><p>Одна самостоятельная мысль<br>с условиями и происхождением.</p></header>
        <div class="mem-assertion-origin mem-from-paper" data-point="paper-result"><span class="mem-label">Из статьи <code>paper_claim</code></span><p><b>Что утверждают авторы</b></p><dl><dt>Источник</dt><dd>Статья + раздел, таблица или теорема</dd><dt>Содержание</dt><dd>Формулировка, тип и условия применимости</dd></dl></div>
        <div class="mem-assertion-origin mem-from-project" data-point="project-result"><span class="mem-label">Из проекта <code>hypothesis</code></span><p><b>Что проверяем мы</b></p><dl><dt>Основания</dt><dd>Метрики и артефакты или доказательство</dd><dt>Оценка</dt><dd>Проверяется → подтверждено, опровергнуто или неопределённо</dd></dl><small>Один ID до и после проверки. Итог требует обоснования.</small></div>
        <footer>Два связанных типа записей.<br>Общие темы, связи и поиск.</footer>
      </article>
      <div class="mem-personal"><span class="mem-label">Тот же агент сохраняет личный контекст</span><div class="mem-personal-stores">
        <article class="mem-obsidian" data-point="obsidian"><span class="mem-store-mark" aria-hidden="true">◇</span><h3>Obsidian</h3><p>Полный разбор, черновики, протоколы и графики в папке своего проекта или библиотеки.</p></article>
        <article class="mem-palace" data-point="mempalace"><span class="mem-store-mark" aria-hidden="true">⌘</span><h3>MemPalace</h3><p>Дневник сессий, исходные фрагменты и решения. Контекст для следующего хода агента.</p></article>
      </div></div>
      <div class="mem-context"><span><b>Решения</b> выбранные действия</span><span><b>Журнал</b> факты и события</span><span><b>Термины</b> определения</span><span><b>Ресурсы</b> код, данные, серверы</span><p>В MCP также сохраняется рабочий контекст. У записей есть ID, источники, авторство и история изменений.</p></div>
    </div>`;
  }

  function ingestion(){
    return `<section class="mem-chapter" aria-labelledby="mem-ingest-title" data-chapter="ingest">
      <header class="mem-chapter-heading"><div><span class="mem-label">Кто и как добавляет знание</span><h2 id="mem-ingest-title">Из статьи. Из проекта.<br><em>Из рабочего разговора.</em></h2></div><p>Материалы приносят люди и агенты. Каждый путь заканчивается записью с источником и чтением сохранённого результата.</p></header>
      <header class="mem-flow-heading"><h3>Статья → утверждения авторов</h3><p>Поиск и рекомендация дают кандидата. В общий корпус попадает выбранная и разобранная работа.</p></header>
      <div class="mem-ingest-map mem-diagram" data-map="ingest">${svg}${feeders('ingest')}
        ${station('document','Материал','Статья и источник',`<div class="mem-paper-sheet"><span>arXiv / DOI</span><b>Полный текст</b><div aria-hidden="true"><i></i><i></i><i></i><i></i></div><small>Таблицы · формулы · приложения</small></div><p>Сначала ищем существующую запись. В личном маршруте Zotero хранит библиографию и PDF.</p>${api('get_paper','upsert_paper')}`)}
        ${station('reader','Смысл выделяет агент','Прочитать и разобрать',`<p>Литературный Hermes читает материал, сверяет числа и предпосылки, отделяет выводы авторов от собственных проверок.</p><div class="mem-model"><span>Модель литературного профиля</span><b>gpt-5.6-sol</b></div><p>Полный разбор остаётся в Obsidian. Другой агент может пройти тот же путь через <code>paper-ingest</code>.</p>`)}
        ${station('sections','Детерминированный перенос','Сохранить разделы',`<div class="mem-section-stack"><span>Метод</span><span>Теория</span><span>Эксперименты</span><span>Ограничения</span></div><p>Парсер переносит текст по заголовкам разбора: название, порядок, содержимое и <code>paper_id</code>.</p><small>Это не нарезка PDF на случайные куски. Имена разделов сохраняются.</small>${api('add_paper_section')}`)}
        ${station('claims','Научный результат разбора','Выделить утверждения',`<p>Агент формулирует каждую независимую мысль своими словами, сохраняя условия.</p><div class="mem-claim-fan"><div><b>Результат</b><span>Метрика, число, модель, данные</span></div><div><b>Теорема</b><span>Предпосылки и полученный вывод</span></div><div><b>Метод / определение</b><span>Что предложено и что означает</span></div></div><small>Число записей задаёт содержание. Квоты нет.</small>${api('record_paper_claim')}`)}
      </div>
      <div class="mem-ingest-end"><div class="mem-claim-schema"><span class="mem-label">Каждая запись</span><p><b>Формулировка</b>${arrow}<b>Тип и условия</b>${arrow}<b>Статья + место</b>${arrow}<b>Устойчивый ID</b></p><small>Цитата необязательна. Ссылка на источник и точное место нужны для проверки.</small></div><div class="mem-claim-relations"><span class="mem-label">Связь с нашей работой</span><p>Утверждение статьи ${arrow} <b>наше утверждение</b></p><small>Предыстория · baseline · поддерживает · противоречит · воспроизводит. Связь добавляет агент, когда она обоснована.</small>${api('link_paper')}</div></div>
      ${projectIngestion()}${checkpointIngestion()}
      <div class="mem-save-route"><h3>Один порядок сохранения</h3><ol><li><b>Найти уже записанное</b><p>По стабильному ID и смыслу. Дополнение сохраняет чужие записи, разделы и связи.</p></li><li><b>Проверить запись</b><p>MCP проверяет права, обязательные поля и связанные ID. Повтор запроса с тем же ключом не создаёт новый объект там, где поддержан <code>idempotency_key</code>.</p></li><li><b>Сохранить и перечитать</b><p>Агент получает запись обратно и сверяет поля. Если запись не прошла, остаётся конкретный pending с причиной.</p></li></ol></div>
    </section>`;
  }

  function projectIngestion(){
    return `<div class="mem-project-flow"><header class="mem-flow-heading"><h3>Проект → наши утверждения и основания</h3><p>Сначала определяется существующий проект. Его материалы разбираются на связанные научные и рабочие записи.</p></header>
      <div class="mem-project-map mem-diagram" data-map="project">${svg}${feeders('project')}
        ${station('workspace','Материалы исследования','Проект и источник',`<div class="mem-project-sheet"><span>PROJECT</span><b>Наше исследование</b><div><span>Код и протоколы</span><span>Логи и таблицы</span><span>Разборы встреч</span></div></div><p>Один project_id связывает работу. Источник указывает, откуда взялся каждый факт.</p>${api('get_project_by_slug','publish_source_note')}`)}
        ${station('claim','Что хотим выяснить','Сформулировать',`<p>Одна проверяемая мысль: формулировка, условия, ожидаемый результат и критерий опровержения.</p><div class="mem-section-stack"><span>Эмпирическое утверждение</span><span>Теоретическое утверждение</span></div><small>Утверждение получает ID в начале и сохраняет его после проверки.</small>${api('create_hypothesis')}`)}
        ${station('grounds','Две формы проверки','Сохранить основания',`<div class="mem-research-branches"><div><b>Эксперимент → измерение</b><p>Протокол и реальный статус прогона; метрики, seeds, артефакты и ограничения.</p>${api('record_experiment','record_evidence')}</div><div><b>Математический вывод</b><p>Предпосылки → выкладка → результат. Отмечается полнота доказательства.</p>${api('record_derivation')}</div></div>`)}
        ${station('assessment','Научный итог','Оценить утверждение',`<p>Агент сопоставляет основания с исходным критерием и проверяет противоречия.</p><div class="mem-verdicts"><b>Подтверждено</b><b>Опровергнуто</b><b>Неопределённо</b></div><small>Итог и обоснование записываются в ту же hypothesis. Пока проверка идёт, её не объявляют законченной.</small>${api('update_hypothesis','get_hypothesis')}`)}
      </div>
      <div class="mem-operational-paths"><span><b>Наблюдение или сбой</b>${arrow}<span>Журнал</span>${api('record_journal')}</span><span><b>Принятый выбор</b>${arrow}<span>Решение + основания</span>${api('propose_decision','update_decision')}</span><span><b>Сервер, данные, квота</b>${arrow}<span>Ресурс</span>${api('upsert_resource')}</span></div>
      <p class="mem-flow-note">Готовые результаты можно добавить к уже существующим записям. Разбор созвона приносит идеи и договорённости; предполагаемый результат остаётся гипотезой до проверки.</p>
    </div>`;
  }

  function checkpointIngestion(){
    return `<div class="mem-checkpoint-flow"><header class="mem-flow-heading"><h3>Hook → сохранение итогов рабочего хода</h3><p>Ещё один вход в ту же базу: сохранить полезное из работы, даже если отдельной команды на добавление не было.</p></header>
      <div class="mem-hook-map mem-diagram" data-map="hook">${svg}
        ${station('session','В ходе работы','Накапливается контекст','<div class="mem-session-lines"><span>Получили результат</span><span>Нашли ограничение</span><span>Выбрали следующий шаг</span></div><p>У Codex, Claude Code и автономных Hermes появляются факты, которые понадобятся следующему агенту.</p>')}
        ${station('checkpoint','Автоматическое напоминание','Срабатывает hook','<div class="mem-cadence"><span><b>Codex / Claude Code</b>после 10 сообщений человека</span><span><b>Автономный Hermes</b>каждые 5 рабочих ходов</span></div><small>Hook просит завершить сохранение. Он не выбирает научный вывод и сам не публикует его.</small>')}
        ${station('curator','Действует агент','Отобрать и записать','<p>Агент читает контекст, выбирает долговечные результаты, проверяет существующие записи и тип каждого факта.</p><small>В общий MCP идут результаты в пределах разрешённой публикации. Если сохранять нечего, лишних записей не создаётся.</small>')}
        ${station('ack','Подтверждение','Прочитать обратно','<div class="mem-checkpoint-stores"><span><b>MCP</b>знание, основания, ID</span><span><b>Obsidian</b>проектные заметки</span><span><b>MemPalace</b>контекст и история</span></div><small>Успешный ответ и чтение записи подтверждают сохранение. При отказе остаётся pending с причиной.</small>')}
      </div><p class="mem-flow-note">В Hermes напоминание приходит в основной ход; проверка перед завершением применяется к ходам с правками. Отложенное лежит в lab-inbox до подтверждённой записи. Ни hook, ни экспорт канбана не заменяют научную оценку агентом.</p>
    </div>`;
  }

  function analysisScenarios(){
    const cases=[
      ['compare','Сравнить результаты','Какая проверка сильнее?',`<b>Эксперименты и метрики</b>${api('get_project_context','get_related')}`,`<b>Агент сверяет протоколы</b><small>Данные, baseline, seeds, единицы и ограничения</small>`,`<b>Сравнение и графики</b><small>results-analysis · academic-plotting, если исходные данные доступны</small>`, 'Числа сравниваются в одинаковых условиях. Отсутствующие повторы и метрики остаются видимыми пробелами.'],
      ['assess','Проверить вывод','Что уже подтверждено?',`<b>Утверждение и критерий</b>${api('get_hypothesis')}`,`<b>Измерения или доказательство</b>${api('get_related','list_derivations')}`,`<b>Оценка с основаниями</b><small>Агент объясняет статус и границы вывода</small>`, 'При новой обоснованной оценке агент обновляет тот же ID через update_hypothesis и читает результат обратно.'],
      ['literature-check','Сопоставить с литературой','Где наши результаты расходятся?',`<b>Наше утверждение</b>${api('find_related_papers')}`,`<b>Тезисы и источники статьи</b>${api('get_paper','list_paper_claims')}`,`<b>Сходство или противоречие</b><small>С учётом данных, предпосылок и области применимости</small>`, 'Поиск предлагает соседние работы. Содержательную связь устанавливает агент после чтения; похожий текст не доказывает поддержку.'],
      ['gaps','Найти незакрытое','Чего не хватает для вывода?',`<b>Состояние базы</b>${api('lab_health')}`,`<b>Проверить найденный пробел</b><small>Нет проверки, evidence, числовых metrics или опоры решения</small>`,`<b>Понятный следующий шаг</b><small>Что записать, уточнить или проверить</small>`, 'Проверка структуры выявляет пропуски. Она не объявляет гипотезу ложной и не оценивает научное качество автоматически.'],
      ['resume','Продолжить работу','Что изменилось с прошлого раза?',`<b>Новые записи</b>${api('recent_changes')}`,`<b>История и контекст</b>${api('get_timeline','get_project_context')}`,`<b>Сводка изменений</b><small>Новые результаты, решения и оставшиеся вопросы</small>`, 'Агент собирает сводку из записанной истории и открытых вопросов проекта, сохраняя ссылки на основания.']
    ];
    return `<div class="mem-analysis-scenarios"><header class="mem-flow-heading"><h3>После поиска — разобраться в результате</h3><p>MCP возвращает записи и связи. Исследователь или агент проверяет смысл, сравнивает данные и формулирует вывод.</p></header>${cases.map(([id,label,title,a,b,c,note])=>`<article class="mem-analysis-case" data-scenario="${id}"><div><span class="mem-label">${label}</span><h4>${title}</h4></div><div class="mem-walk-path"><span>${a}</span>${arrow}<span>${b}</span>${arrow}<span>${c}</span></div><p>${note}</p></article>`).join('')}<div class="mem-useful-context"><span><b>Кто занимается темой</b>${api('who_works_on_what')}</span><span><b>Какие ресурсы доступны</b>${api('list_lab_resources')}</span><span><b>Что означают термины проекта</b>${api('list_terms')}</span></div></div>`;
  }

  function retrieval(){
    return `<section class="mem-chapter" aria-labelledby="mem-search-title" data-chapter="search">
      <header class="mem-chapter-heading"><div><span class="mem-label">Работа с накопленным знанием</span><h2 id="mem-search-title">Найти. Сопоставить.<br><em>Понять, что следует из результатов.</em></h2></div><p>Поиск собирает материал. Дальше можно проверить утверждение, сравнить эксперименты, найти пробелы или продолжить чужую работу с её контекстом.</p></header>
      <div class="mem-readers"><span class="mem-label">Кто обращается к базе</span><span>Исследователь</span><span>Codex / Claude Code</span><span>Литературный Hermes</span><span>Экспериментальный Hermes</span><small>Вопрос или задача → инструменты MCP → записи с источниками</small></div>
      <div class="mem-search-map mem-diagram" data-map="search">${svg}
        ${station('query','Есть вопрос','Поиск по запросу',`<blockquote>«Что известно о методе и его ограничениях?»</blockquote><p>Вся база, только наша работа или только библиотека. Можно выбрать тип записи.</p>${api('search_lab')}`)}
        ${station('words','Лексика','По словам','<p>PostgreSQL ищет слова и их формы. Русский, английский и словарь терминов помогают сопоставить названия.</p>')}
        ${station('vectors','Семантика','По смыслу','<p>multilingual-E5-large превращает вопрос и тексты в векторы. Косинусная близость находит близкий смысл при разных словах.</p>')}
        ${station('merge','Два корпуса','Объединить','<p>Статьи и наши записи попадают в общий набор. Повторы убираются; совпадение в двух ветках усиливает результат.</p><small>RRF разрешает равенство рангов.</small>')}
        ${station('graph','Контекст','Добавить связи','<p>К найденным записям добавляются связанные проверки, измерения и решения.</p><small>До 3 исходных записей, 2 соседей на каждую, 6 добавлений.</small>')}
        ${station('rerank','Общий порядок','Уточнить релевантность','<p>Jina сравнивает вопрос с каждым кандидатом после добавления связей.</p><div class="mem-model"><span>Cross-encoder</span><b>Jina reranker v2</b><small>multilingual · INT8</small></div>')}
        ${station('answer','Результат','Запись с основанием','<ul class="mem-answer-fields"><li>ID и тип</li><li>Формулировка / фрагмент</li><li>Источник и проект</li><li>Связанные записи</li></ul><p>Агент читает полную запись и отвечает со ссылками. Балл поиска показывает релевантность, а не истинность.</p>')}
      </div>
      <div class="mem-search-settings"><p><b>Режимы:</b> hybrid объединяет слова и смысл; lexical и semantic оставляют одну из веток. Фильтр конкретного проекта исключает общую библиотеку.</p><p><b>Сейчас:</b> Jina INT8, 16 потоков; до 60 кандидатов, до 400 символов на документ. Если E5 недоступна, остаётся лексика; без reranker сохраняется прежний порядок.</p></div>
      ${analysisScenarios()}
      <div class="mem-walks"><h3>Начать с темы, проекта или знакомой записи</h3>
        <div class="mem-walk" data-route="theme"><div><span class="mem-label">Изучить направление</span><h4>От темы к работам</h4></div><div class="mem-walk-path"><span><b>Темы лаборатории</b>${api('list_themes')}</span>${arrow}<span><b>Контекст темы</b>${api('get_theme_context')}</span>${arrow}<span><b>Работы и утверждения</b><small>Своя наука + литература</small></span></div></div>
        <div class="mem-walk" data-route="library"><div><span class="mem-label">Осмотреть библиотеку</span><h4>От раздела к статье</h4></div><div class="mem-walk-path"><span><b>Раздел → папка</b>${api('library_tree')}</span>${arrow}<span><b>Подтема → статьи</b><small>Аннотации помогают выбрать ветку</small></span>${arrow}<span><b>Статья → утверждения</b>${api('get_paper','list_paper_claims')}</span></div><p>Дерево читается напрямую: без векторов и ранжирования. Статьи без подтемы тоже видны.</p></div>
        <div class="mem-walk" data-route="known"><div><span class="mem-label">Есть конкретный объект</span><h4>От проекта или ID</h4></div><div class="mem-walk-path"><span><b>Проект</b>${api('get_project_by_slug')}</span>${arrow}<span><b>Сводка → нужный раздел</b>${api('get_project_context')}</span>${arrow}<span><b>Полная запись</b>${api('get_hypothesis','get_paper')}</span></div><p>Сначала компактная сводка, затем выбранные записи. Не нужно загружать всю историю проекта.</p></div>
        <div class="mem-walk" data-route="evidence"><div><span class="mem-label">Проверить основание</span><h4>От утверждения к проверке</h4></div><div class="mem-walk-path"><span><b>Известная запись</b><small>Устойчивый ID</small></span>${arrow}<span><b>Проверки и источники</b>${api('get_related')}</span>${arrow}<span><b>История изменений</b>${api('get_timeline')}</span></div><p>Это явные сохранённые связи. Их происхождение можно проверить, а историю изменения статуса — прочитать.</p></div>
        <div class="mem-walk" data-route="neighbors"><div><span class="mem-label">Расширить вопрос</span><h4>Найти соседние идеи</h4></div><div class="mem-neighbor-paths"><span><b>Общие термины</b>${api('related_by_terms')}<small>Тематическое соседство, не доказанная связь</small></span><span><b>Литература вокруг гипотезы</b>${api('find_related_papers')}<small>По формулировке и механизму</small></span><span><b>Похожая запись до добавления</b>${api('find_similar')}<small>Кандидаты на дубль; решение за агентом</small></span></div></div>
      </div>
      <div class="mem-vector-cache"><div><span class="mem-label">Как подготовлен смысловой поиск</span><h3>Векторы переиспользуются</h3><p>Ключ: модель + ID записи + хэш текста. Изменился текст — нужен новый вектор.</p></div><div class="mem-cache-path"><span><b>Текст</b><small>Прогрев или новый запрос</small></span>${arrow}<span><b>E5</b><small>Вектор записи</small></span>${arrow}<span><b>PostgreSQL</b><small>retrieval_embeddings</small></span>${arrow}<span><b>Кэш в памяти</b><small>Повторное использование</small></span></div><p>Готовый вектор берётся из памяти или базы. <code>warm_all.py</code> считает их заранее; запрос досчитывает до 16 недостающих на корпус. Для вопросов об устройстве лаборатории поиск отдаёт проекты, разделы и подтемы, сохраняя семантику и reranker, но пропуская обход исследовательских связей.</p></div>
    </section>`;
  }

  function render(p, processes){
    return `<div class="process-shell memory-observatory" style="--accent:#b9ff66">
      <header class="mem-header"><div><button class="back-button" data-back type="button">← Весь Atlas</button><span class="mem-coordinate">BRAIn Lab / Lab Knowledge MCP</span></div><h1 tabindex="-1">Как работа становится <em>знанием</em></h1></header>
      <section class="mem-first-map" aria-label="Общая карта: источники, утверждения и личная память" data-chapter="overview">${overview(processes)}<div class="mem-practice-notes"><p><b>Когда сохраняется:</b> по ходу работы и при checkpoint. Hook напоминает, агент выбирает содержание, записывает и читает результат обратно.</p><p><b>Где задачи:</b> общие в Yonote, личные в Operon. Канбан Hermes остаётся у Hermes; Python обновляет его просмотр в папке проекта Obsidian.</p></div></section>
      ${ingestion()}${retrieval()}
      <footer class="mem-footer"><p>Схемы показывают устройство системы, без данных частной базы. Модели и параметры сверены 10-09-2026.</p><a href="#mcp-live">Перейти к записям MCP ↗</a><details class="mem-tools"><summary>Навыки и инструменты этого слоя</summary><div>${p.tools.map(t=>`<button type="button" data-skill="${esc(t.id)}">${esc(t.title)}</button>`).join('')}</div></details><a href="https://github.com/Vepricov/claude-brainlab/blob/main/docs/knowledge-base.md" target="_blank" rel="noopener noreferrer">Контракт и исходники ↗</a></footer>
      <dialog class="skill-dialog" aria-labelledby="skill-dialog-title"></dialog>
    </div>`;
  }

  const edges={
    overview:[['writer','paper','paper'],['writer','question','project'],['paper','extract','paper'],['question','check','project'],['extract','paper-result','paper'],['check','project-result','project']],
    ingest:[['document','reader','paper'],['reader','sections','paper'],['sections','claims','paper']],
    project:[['workspace','claim','project'],['claim','grounds','project'],['grounds','assessment','project']],
    hook:[['session','checkpoint','violet'],['checkpoint','curator','violet'],['curator','ack','project']],
    search:[['query','words','paper'],['query','vectors','violet'],['words','merge','paper'],['vectors','merge','violet'],['merge','graph','project'],['graph','rerank','project'],['rerank','answer','project']]
  };
  let observer;
  function draw(map){
    const box=map.getBoundingClientRect();if(!box.width||!box.height)return;
    const svg=map.querySelector('.mem-connections'),kind=map.dataset.map,compact=matchMedia('(max-width:760px)').matches;
    const bounds=id=>map.querySelector(`[data-point="${id}"]`).getBoundingClientRect();
    const path=(from,to,color)=>{
      const a=bounds(from),b=bounds(to);let d;
      if(kind==='overview'&&from==='writer'){
        if(matchMedia('(max-width:1100px)').matches)return '';
        const rail=b.left-box.left-16,ay=a.top-box.top+11,bx=b.left-box.left+4,by=b.top-box.top+35;
        d=`M${a.left-box.left+4},${ay} H${rail} V${by} H${bx}`;
      }else if(b.left>=a.right-3){
        const node=map.querySelector(`[data-point="${from}"]`),heading=node.querySelector('h3')?.getBoundingClientRect();
        const ax=(heading?Math.min(a.right,heading.right+12):a.right)-box.left,ay=a.top-box.top+35,bx=b.left-box.left-2,by=b.top-box.top+(to.endsWith('result')?23:35),mid=(ax+bx)/2;
        d=`M${ax},${ay} C${mid},${ay} ${mid},${by} ${bx},${by}`;
      }else{
        const ax=a.left-box.left+12,ay=a.bottom-box.top+5,bx=b.left-box.left+12,by=b.top-box.top-8;
        const rail=compact?3:Math.min(ax,bx)-17;
        d=`M${ax},${ay} H${rail} V${by-6} Q${rail},${by} ${rail+6},${by} H${bx}`;
      }
      return `<path data-edge="${from}:${to}" class="mem-line-${color}" d="${d}" marker-end="url(#mem-arrow-${kind}-${color})"/>`;
    };
    let lines=edges[kind].map(e=>path(...e)).join('');
    if(inlets[kind]){
      const target=kind==='ingest'?'document':'workspace',b=bounds(target),group=map.querySelector('.mem-feeders').getBoundingClientRect(),bx=b.left-box.left+4,by=b.top-box.top+35;
      for(const [id] of inlets[kind]){
        const source=`${kind}-via-${id}`,a=bounds(source),ax=a.left-box.left+4,ay=a.top-box.top+8,rail=compact?group.left-box.left-10:group.bottom-box.top+20;
        const d=compact?`M${ax},${ay} H${rail} V${by} H${bx}`:`M${ax},${ay+8} V${rail-6} Q${ax},${rail} ${Math.max(bx,ax-6)},${rail} H${bx} V${by}`;
        lines+=`<path data-edge="${source}:${target}" class="mem-line-${kind==='ingest'?'paper':'project'} mem-line-feeder" d="${d}" marker-end="url(#mem-arrow-${kind}-${kind==='ingest'?'paper':'project'})"/>`;
      }
    }
    if(kind==='overview'&&matchMedia('(min-width:1101px)').matches){
      const b=bounds('writer');map.querySelectorAll('[data-origin]').forEach(el=>{const a=el.getBoundingClientRect(),ax=a.right-box.left,ay=a.top-box.top+a.height/2,bx=b.left-box.left+4,by=b.top-box.top+11,mid=(ax+bx)/2;lines+=`<path class="mem-line-input" d="M${ax},${ay} C${mid},${ay} ${mid},${by} ${bx},${by}"/>`});
      for(const id of ['obsidian','mempalace']){
        const a=bounds(id),rail=b.left-box.left-16,x=a.left-box.left+7,y=a.top-box.top;
        lines+=`<path data-edge="writer:${id}" class="mem-line-private" d="M${b.left-box.left+4},${b.top-box.top+11} H${rail} V${y-6} H${x} V${y+10}"/>`;
      }
    }
    const colors={paper:'#9ec9d8',project:'#c5df94',violet:'#bba8e1'};
    svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
    svg.innerHTML=`<defs>${Object.entries(colors).map(([c,fill])=>`<marker id="mem-arrow-${kind}-${c}" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="${fill}"/></marker>`).join('')}</defs>${lines}`;
  }
  function bind(root){
    observer?.disconnect();const scope=root.querySelector('.memory-observatory');if(!scope)return;
    const maps=[...scope.querySelectorAll('.mem-diagram')];
    observer=new ResizeObserver(()=>maps.forEach(draw));maps.forEach(m=>observer.observe(m));
    requestAnimationFrame(()=>maps.forEach(draw));
  }
  return {render,bind};
})();
