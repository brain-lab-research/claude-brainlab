"""Отправка приглашения: без ключа она обязана сказать об этом, а не притвориться."""

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mailer  # noqa: E402


def test_without_a_key_it_says_so_instead_of_pretending(monkeypatch) -> None:
    """Молчаливый пропуск хуже отказа: человек ждёт письма, которого никто не отправлял."""
    monkeypatch.delenv("LAB_MAIL_API_KEY", raising=False)
    monkeypatch.delenv("LAB_MAIL_KEY_FILE", raising=False)

    with pytest.raises(mailer.MailNotConfigured, match="ключа почтовой службы нет"):
        mailer.send_invite("student@lab", "https://example.org/invite/x")


def test_a_key_in_a_file_is_read(monkeypatch, tmp_path) -> None:
    """Ключ живёт в файле с правами 600, а не в командной строке, где его видно в списке процессов."""
    key_file = tmp_path / "mail-key"
    key_file.write_text("ключ-из-файла\n", encoding="utf-8")
    monkeypatch.delenv("LAB_MAIL_API_KEY", raising=False)
    monkeypatch.setenv("LAB_MAIL_KEY_FILE", str(key_file))

    assert mailer._key() == "ключ-из-файла"


def test_the_letter_carries_the_link_and_says_one_key_is_enough() -> None:
    """Письмо читает человек, который про базу ещё ничего не знает: лишних понятий в нём быть не должно."""
    letter = mailer.LETTER.format(link="https://example.org/invite/abc",
                                  attachment="С чего начать.md", repo="")

    assert "https://example.org/invite/abc" in letter
    assert "одноразовая" in letter
    assert "работает сразу для всех ваших агентов" in letter
    assert "пароль" not in letter.lower(), "пароля в этой схеме нет вовсе"
    assert max(len(line) for line in letter.splitlines()) > 100, (
        "абзацы не ломаются переносами: их переносит почтовый клиент, а не мы"
    )


def test_a_refusal_from_the_service_is_passed_through(monkeypatch, tmp_path) -> None:
    """Причину отказа придумывать нельзя: её знает почтовая служба, а не мы."""
    import io
    import urllib.error

    guide = tmp_path / "guide.md"
    guide.write_text("записка", encoding="utf-8")
    monkeypatch.setenv("LAB_GUIDE_FILE", str(guide))
    monkeypatch.setenv("LAB_MAIL_API_KEY", "поддельный")

    def refuse(*_args, **_kwargs):
        # fp=None заставляет HTTPError завести временный файл, и на сборке объекта
        # он падает KeyError'file'. Пустой поток даёт ту же ошибку без этого эффекта.
        raise urllib.error.HTTPError("url", 422, "Unprocessable", {}, io.BytesIO(b""))

    monkeypatch.setattr(mailer.urllib.request, "urlopen", refuse)
    with pytest.raises(mailer.MailFailed, match="422"):
        mailer.send_invite("student@lab", "https://example.org/invite/x")


def test_the_letter_refuses_to_go_out_without_the_guide(monkeypatch, tmp_path) -> None:
    """Голая ссылка без вводной записки отправляет человека в никуда, а он этого не поймёт."""
    monkeypatch.setenv("LAB_MAIL_API_KEY", "поддельный")
    monkeypatch.setenv("LAB_GUIDE_FILE", str(tmp_path / "которой-нет.md"))

    with pytest.raises(mailer.MailFailed, match="вводной записки нет"):
        mailer.send_invite("student@lab", "https://example.org/invite/x")


def test_the_guide_travels_with_the_letter(monkeypatch, tmp_path) -> None:
    """Записка едет вложением, а письмо называет её по имени, чтобы человек её заметил."""
    guide = tmp_path / "С чего начать.md"
    guide.write_text("# Записка\nчто это и как подключиться", encoding="utf-8")
    monkeypatch.setenv("LAB_MAIL_API_KEY", "поддельный")
    monkeypatch.setenv("LAB_GUIDE_FILE", str(guide))
    monkeypatch.setenv("LAB_REPO_URL", "https://github.com/brain-lab-research/тайный")
    sent = {}

    class _Answer:
        def read(self):
            return '{"id": "письмо-1"}'.encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    def capture(request, timeout=0):
        sent["payload"] = json.loads(request.data.decode())
        return _Answer()

    monkeypatch.setattr(mailer.urllib.request, "urlopen", capture)
    assert mailer.send_invite("student@lab", "https://example.org/invite/x") == "письмо-1"

    payload = sent["payload"]
    assert payload["attachments"][0]["filename"] == "С чего начать.md"
    assert "Записка" in base64.b64decode(payload["attachments"][0]["content"]).decode()
    assert "С чего начать.md" in payload["text"], "письмо называет вложение"
    assert "github.com/brain-lab-research/тайный" in payload["text"], "и даёт ссылку на код"


def test_without_a_repo_the_letter_does_not_mention_one(monkeypatch, tmp_path) -> None:
    """Пустая строка про репозиторий в письме выглядит как забытая переменная."""
    guide = tmp_path / "guide.md"
    guide.write_text("записка", encoding="utf-8")
    monkeypatch.setenv("LAB_GUIDE_FILE", str(guide))
    monkeypatch.delenv("LAB_REPO_URL", raising=False)

    letter = mailer.LETTER.format(link="ссылка", attachment="guide.md", repo="")

    assert "github" not in letter.lower()
