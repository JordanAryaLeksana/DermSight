import smtplib
from email.message import EmailMessage


class EmailConfigurationError(RuntimeError):
    pass


class EmailService:
    def __init__(self, config):
        self.host = config["SMTP_HOST"]
        self.port = config["SMTP_PORT"]
        self.username = config["SMTP_USERNAME"]
        self.password = config["SMTP_PASSWORD"]
        self.sender = config["SMTP_FROM"]
        self.recipient = config["ASPIRATION_EMAIL_TO"]
        self.use_tls = config["SMTP_USE_TLS"]
        self.timeout = config["SMTP_TIMEOUT"]

    def send_aspiration(self, data):
        if not all((self.host, self.sender, self.recipient)):
            raise EmailConfigurationError("Layanan email aspirasi belum dikonfigurasi.")

        message = EmailMessage()
        message["Subject"] = f"Aspirasi Baru DermSight — {data['kind']}"
        message["From"] = self.sender
        message["To"] = self.recipient
        message.set_content(
            "Aspirasi Baru dari DermSight\n\n"
            f"Jenis:\n{data['kind']}\n\n"
            f"Nama:\n{data['name'] or '(anonim)'}\n\n"
            f"Kontak:\n{data['contact'] or '(tidak diberikan)'}\n\n"
            f"Pesan:\n{data['message']}\n"
        )

        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(message)
