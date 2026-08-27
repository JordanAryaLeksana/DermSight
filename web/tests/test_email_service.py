from web.app.services.email_service import EmailService


def test_email_service_uses_configured_recipient(monkeypatch):
    sent = []
    class FakeSMTP:
        def __init__(self, host, port, timeout): assert host == "smtp.test"
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def starttls(self): pass
        def login(self, username, password): assert username == "user"
        def send_message(self, message): sent.append(message)
    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    config = {"SMTP_HOST": "smtp.test", "SMTP_PORT": 587, "SMTP_USERNAME": "user", "SMTP_PASSWORD": "secret", "SMTP_FROM": "from@test", "ASPIRATION_EMAIL_TO": "to@test", "SMTP_USE_TLS": True, "SMTP_TIMEOUT": 5}
    EmailService(config).send_aspiration({"kind": "Kritik", "name": "", "contact": "", "message": "Pesan pengujian"})
    assert len(sent) == 1
    assert sent[0]["To"] == "to@test"
    assert "Pesan pengujian" in sent[0].get_content()
