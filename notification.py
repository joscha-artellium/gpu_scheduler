"""Send notification emails over SMTP, configured from a .env file.

Credentials come from ./email_credentials.env if present, else from
~/.config/mailnotify/email_credentials.env. The home directory is that of the
user who invoked sudo ($SUDO_USER), not $HOME — sudo may or may not reset $HOME
to /root depending on the sudoers `always_set_home` setting, so one 0600 file in
your own config directory serves both plain and sudo invocations.

Real environment variables always win over the file.
"""

import html
import logging
import os
import pwd
import smtplib
from collections.abc import Sequence
from email import utils
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = "mailnotify"
CREDENTIALS_NAME = "email_credentials.env"
SMTP_TIMEOUT = 20.0


def credentials_path() -> Path | None:
    """cwd first, then the config dir of the human who ran the command."""
    local = Path.cwd() / CREDENTIALS_NAME
    if local.is_file():
        return local
    try:
        if sudo_user := os.environ.get("SUDO_USER"):
            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        else:
            home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        return None
    personal = home / ".config" / APP_DIR / CREDENTIALS_NAME
    return personal if personal.is_file() else None


def send_email(
    subject: str,
    content: str,
    to_address: Sequence[str] | None = None,
    *,
    raise_unconfigured: bool = True,
) -> bool:
    """Return True only if the message was handed to the SMTP server."""
    logger = logging.getLogger(__name__)
    credentials = credentials_path()
    if credentials:
        load_dotenv(credentials, override=False)
    origin = str(credentials) if credentials else "the environment"

    if to_address is None:
        env_to = os.getenv("MAIL_TO")
        if not env_to:
            logger.error(f"No recipient provided and MAIL_TO missing in {origin}")
            return False
        to_address = [address.strip() for address in env_to.split(",")]

    password = os.environ.get("MAIL_PASSWORD")
    if password is None:
        if raise_unconfigured:
            err_msg = f"Environment variable MAIL_PASSWORD not set (checked {origin})"
            raise KeyError(err_msg)
        logger.error(f"No email password provided in {origin}")
        return False

    smtp_server = os.environ.get("MAIL_SMTP_SERVER", "smtp.gmail.com")
    sender_email = os.environ.get("MAIL_FROM", "email.update.notify2@gmail.com")
    smtp_port = int(os.environ.get("MAIL_SMTP_PORT", "587"))

    msg = EmailMessage()
    # Newlines in a header would let content forge further headers
    msg["Subject"] = " ".join(subject.splitlines())
    # Spam filters prefer "Friendly Name <address@domain.com>" formatting
    msg["From"] = utils.formataddr(("Automated Notifier", sender_email))
    msg["To"] = ", ".join(to_address)
    msg["Date"] = utils.formatdate(localtime=True)
    msg["Message-ID"] = utils.make_msgid(domain=sender_email.split("@")[-1])
    msg.set_content(content)
    # <pre> + escaping: clients prefer the HTML part, and <p> would collapse the
    # whitespace that log tails and fixed-width tables depend on
    msg.add_alternative(
        f"<html><body><pre>{html.escape(content)}</pre></body></html>", subtype="html"
    )

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=SMTP_TIMEOUT) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg, from_addr=sender_email, to_addrs=list(to_address))
        logger.info(f"Email sent successfully to: {msg['To']}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.exception("Authentication failed. Check your email and App Password")
    except smtplib.SMTPRecipientsRefused:
        logger.exception("The server refused the recipient addresses")
    except smtplib.SMTPException:
        logger.exception("An SMTP protocol error occurred")
    except (ConnectionError, TimeoutError, OSError):
        logger.exception("A network error occurred while connecting to the mail server")
    return False
