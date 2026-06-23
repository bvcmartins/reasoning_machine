"""Authentication configuration for the Reasoning Machine Streamlit app.

Mirrors personal_assistant's auth: streamlit-authenticator 0.4.x, credentials
from env vars, a pre-hashed admin password. Only the cookie name differs
(`reasoning_machine_cookie`) so the two apps don't share a session cookie.
"""
import os
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv

# Load environment variables from project root .env
load_dotenv(Path(__file__).resolve().parent.parent / '.env')


def get_authenticator():
    """Create and return the authenticator instance."""
    username = os.getenv('ADMIN_USERNAME', 'admin')
    password_hash = os.getenv('ADMIN_PASSWORD_HASH', '').strip("'\"") or None
    cookie_key = os.getenv('COOKIE_KEY', 'default_cookie_key_change_me_in_production')

    if not password_hash:
        st.error("ADMIN_PASSWORD_HASH not set in environment variables")
        st.stop()

    credentials = {
        'usernames': {
            username: {
                'name': 'Administrator',
                'password': password_hash,
            }
        }
    }

    return stauth.Authenticate(
        credentials,
        'reasoning_machine_cookie',  # Cookie name (distinct from personal_assistant)
        cookie_key,                  # Cookie key for signing
        cookie_expiry_days=7,
        auto_hash=False,             # Password is already hashed
    )


def require_authentication():
    """Require authentication to proceed. Returns True if authenticated."""
    authenticator = get_authenticator()
    authenticator.login()

    if st.session_state.get('authentication_status') is False:
        st.error('Username/password is incorrect')
        return False
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')
        return False
    else:
        name = st.session_state.get('name', 'User')
        with st.sidebar:
            st.write(f'Welcome, *{name}*')
            authenticator.logout('Logout', 'sidebar')
        return True
