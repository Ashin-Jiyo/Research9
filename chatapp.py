import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from google import genai
from google.genai import types


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set the GOOGLE_API_KEY environment variable before starting the server."
    )

client = genai.Client(api_key=API_KEY)

users: Dict[str, Dict[str, str]] = {}
conversations: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
user_seen: Dict[str, Dict[Tuple[str, str], int]] = {}


def current_user() -> Optional[str]:
    username = session.get("username")
    if username and username in users:
        return username
    session.pop("username", None)
    return None


def conversation_key(user_a: str, user_b: str) -> Tuple[str, str]:
    return tuple(sorted((user_a, user_b)))


def translate_message(message_text: str, target_language: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=message_text,
        config=types.GenerateContentConfig(
            system_instruction=(
                f"Translate this message to {target_language}. "
                "Only return the translated text."
            ),
            temperature=0.6,
        ),
    )

    return response.text.strip() if response.text else message_text


def ensure_user_seen(username: str) -> Dict[Tuple[str, str], int]:
    return user_seen.setdefault(username, {})


def mark_conversation_seen(viewer: str, partner_key: str) -> None:
    key = conversation_key(viewer, partner_key)
    ensure_user_seen(viewer)[key] = len(conversations.get(key, []))


def unread_count(username: str, partner_key: str) -> int:
    key = conversation_key(username, partner_key)
    total = len(conversations.get(key, []))
    seen = ensure_user_seen(username).get(key, 0)
    return max(total - seen, 0)


def save_message(sender: str, recipient: str, original_text: str, translated_text: str) -> Dict[str, str]:
    key = conversation_key(sender, recipient)
    entry = {
        "sender": sender,
        "recipient": recipient,
        "original_text": original_text,
        "translated_text": translated_text,
    }
    bucket = conversations.setdefault(key, [])
    bucket.append(entry)
    ensure_user_seen(sender)[key] = len(bucket)
    return entry


def serialize_messages(viewer: str, partner_key: str, mark_seen: bool = False) -> List[Dict[str, str]]:
    key = conversation_key(viewer, partner_key)
    raw_messages = conversations.get(key, [])
    partner_data = users[partner_key]
    messages_for_view: List[Dict[str, str]] = []

    for entry in raw_messages:
        is_self = entry["sender"] == viewer
        if is_self:
            primary_text = entry["original_text"]
            secondary_label = (
                f"Translated for {partner_data['display_name']}"
                f" ({partner_data['language']}):"
            )
            secondary_text = entry["translated_text"]
        else:
            primary_text = entry["translated_text"] or entry["original_text"]
            secondary_label = (
                f"Original message in {users[entry['sender']]['language']}:"
            )
            secondary_text = entry["original_text"]

        messages_for_view.append(
            {
                "is_self": is_self,
                "sender_label": "You" if is_self else users[entry["sender"]]["display_name"],
                "primary_text": primary_text,
                "secondary_label": secondary_label,
                "secondary_text": secondary_text,
                "created_by": entry["sender"],
            }
        )

    if mark_seen:
        mark_conversation_seen(viewer, partner_key)

    return messages_for_view


def build_dashboard_payload(user: str, query: str) -> Dict[str, List[Dict[str, str]]]:
    query_lower = query.strip().lower()

    matches: List[Dict[str, str]] = []
    for username, data in users.items():
        if username == user:
            continue
        if not query_lower or query_lower in data["display_name"].lower():
            matches.append(
                {
                    "username": username,
                    "display_name": data["display_name"],
                    "language": data["language"],
                    "unread_count": unread_count(user, username),
                }
            )
    matches.sort(key=lambda item: item["display_name"].lower())

    recent_contacts: List[Dict[str, str]] = []
    seen_partners: set[str] = set()
    for key in conversations.keys():
        if user in key:
            other = key[0] if key[0] != user else key[1]
            if other in users and other not in seen_partners:
                seen_partners.add(other)
                data = users[other]
                recent_contacts.append(
                    {
                        "username": other,
                        "display_name": data["display_name"],
                        "language": data["language"],
                        "unread_count": unread_count(user, other),
                    }
                )
    recent_contacts.sort(key=lambda item: item["display_name"].lower())

    return {"matches": matches, "recent_contacts": recent_contacts}


@app.route("/", methods=["GET", "POST"])
def login():
    user = current_user()
    if user:
        return redirect(url_for("dashboard"))

    error = None
    username_value = ""
    language_value = ""

    if request.method == "POST":
        username_value = request.form.get("username", "").strip()
        language_value = request.form.get("language", "").strip()

        if not username_value:
            error = "Please enter a username."
        else:
            username_key = username_value.lower()
            record = users.get(username_key)

            if record:
                record["display_name"] = username_value
                if language_value:
                    record["language"] = language_value
                ensure_user_seen(username_key)
                session["username"] = username_key
                return redirect(url_for("dashboard"))

            if not language_value:
                error = "Pick a preferred language for your first login."
            else:
                users[username_key] = {
                    "display_name": username_value,
                    "language": language_value,
                }
                ensure_user_seen(username_key)
                session["username"] = username_key
                return redirect(url_for("dashboard"))

    return render_template(
        "login.html",
        error=error,
        username_value=username_value,
        language_value=language_value,
    )


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    query = request.args.get("q", "")
    data = build_dashboard_payload(user, query)

    return render_template(
        "dashboard.html",
        current_user=users[user],
        query=query,
        initial_data=data,
        data_url=url_for("dashboard_data"),
    )


@app.route("/dashboard/data")
def dashboard_data():
    user = current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    query = request.args.get("q", "")
    return jsonify(build_dashboard_payload(user, query))


@app.route("/chat/<partner>")
def chat(partner: str):
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    partner_key = partner.lower()
    if partner_key not in users or partner_key == user:
        return redirect(url_for("dashboard"))

    messages_for_view = serialize_messages(user, partner_key, mark_seen=True)

    return render_template(
        "chat.html",
        current_user=users[user],
        partner_username=partner_key,
        partner=users[partner_key],
        messages=messages_for_view,
        send_url=url_for("send_message", partner=partner_key),
        fetch_url=url_for("chat_messages", partner=partner_key),
    )


@app.route("/chat/<partner>/messages")
def chat_messages(partner: str):
    user = current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    partner_key = partner.lower()
    if partner_key not in users or partner_key == user:
        return jsonify({"error": "Invalid conversation"}), 400

    return jsonify({"messages": serialize_messages(user, partner_key, mark_seen=True)})


@app.route("/chat/<partner>/send", methods=["POST"])
def send_message(partner: str):
    user = current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    partner_key = partner.lower()
    if partner_key not in users or partner_key == user:
        return jsonify({"error": "Invalid conversation"}), 400

    data = request.get_json(silent=True) or {}
    message_text = (data.get("message") or "").strip()
    if not message_text:
        return jsonify({"error": "Type a message before sending."}), 400

    try:
        translated_text = translate_message(
            message_text, users[partner_key]["language"]
        )
        save_message(user, partner_key, message_text, translated_text)
        return jsonify({"message": serialize_messages(user, partner_key)[-1]})
    except Exception as exc:
        return jsonify({"error": f"Could not translate your message: {exc}"}), 500


@app.route("/settings", methods=["GET", "POST"])
def settings():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    error = None
    success = None
    if request.method == "POST":
        new_language = request.form.get("language", "").strip()
        if not new_language:
            error = "Please enter a language before saving."
        else:
            users[user]["language"] = new_language
            success = "Language updated."

    return render_template(
        "settings.html",
        current_user=users[user],
        error=error,
        success=success,
    )


if __name__ == "__main__":
    app.run(debug=True)
