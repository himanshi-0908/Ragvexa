import re

from sqlalchemy.exc import IntegrityError
import streamlit_authenticator as stauth

from utils.database import SessionLocal, User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def get_credentials():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        credentials = {"usernames": {}}
        for u in users:
            credentials["usernames"][u.username] = {
                "email": u.email,
                "name": u.name,
                "password": u.hashed_password,
                "user_id": u.id
            }
        return credentials
    finally:
        db.close()


def register_user(username, email, name, plain_password):
    username = (username or "").strip()
    email = (email or "").strip().lower()
    name = (name or "").strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address"

    if len(plain_password or "") < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return False, "Username already exists"

        if db.query(User).filter(User.email == email).first():
            return False, "An account with this email already exists"

        hashed_password = stauth.Hasher().hash(plain_password)
        new_user = User(
            username=username,
            email=email,
            name=name,
            hashed_password=hashed_password
        )
        db.add(new_user)
        try:
            db.commit()
        except IntegrityError:
            # Two requests raced past the checks above and both tried to
            # insert the same username/email — the unique constraint on
            # the columns is the real guard; this just gives a clean error.
            db.rollback()
            return False, "Username or email already exists"
        return True, "Success"
    finally:
        db.close()
