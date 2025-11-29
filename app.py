import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from openai import OpenAI
from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression

from database import engine, Base, SessionLocal
from models import Transaction

# ---------------------------
# Init DB + App + OpenAI
# ---------------------------

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
app.secret_key = "super-secret-key-change-later"  # change in production

# Load env vars for OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not set. AI assistant will not work.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------
# Helpers: File → DataFrame
# ---------------------------

def read_file_to_df(file_storage):
    """
    Accepts uploaded file (CSV or Excel) and returns pandas DataFrame.
    We normalize column names to lowercase + underscores.
    """
    filename = file_storage.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file_storage)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    # normalize column names: "Order Date" -> "order_date"
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------
# Helpers: Clean + Save to DB
# ---------------------------

def save_transactions_from_df(df: pd.DataFrame):
    """
    Cleans the dataframe and saves rows into the database.
    Tries to map flexible column names to: date, type, amount, category,
    description, customer, product.
    """
    col_map = {}

    # candidate names for each required field
    date_candidates = ["date", "order_date", "transaction_date"]
    type_candidates = ["type", "txn_type", "transaction_type"]
    amount_candidates = ["amount", "price", "selling_price", "revenue", "value", "total"]

    for c in df.columns:
        if c in date_candidates and "date" not in col_map:
            col_map["date"] = c
        if c in type_candidates and "type" not in col_map:
            col_map["type"] = c
        if c in amount_candidates and "amount" not in col_map:
            col_map["amount"] = c

    required_cols = {"date", "type", "amount"}
    if not required_cols.issubset(set(col_map.keys())):
        raise ValueError(
            f"File must contain date, type, amount (or equivalent). Mapped columns: {col_map}"
        )

    # build a normalized df
    out = pd.DataFrame()
    out["date"] = df[col_map["date"]]
    out["type"] = df[col_map["type"]]
    out["amount"] = df[col_map["amount"]]

    # optional columns (if present)
    out["category"] = df["category"] if "category" in df.columns else ""
    out["description"] = df["description"] if "description" in df.columns else ""
    out["customer"] = df["customer"] if "customer" in df.columns else ""
    out["product"] = df["product"] if "product" in df.columns else ""

    # Clean & convert
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["amount"])

    out["type"] = out["type"].fillna("").str.lower()

    # small helper: map common synonyms
    type_mapping = {
        "income": "income",
        "in": "income",
        "credit": "income",
        "sale": "income",
        "sales": "income",
        "revenue": "income",
        "earning": "income",
        "earnings": "income",
        "expense": "expense",
        "ex": "expense",
        "debit": "expense",
        "cost": "expense",
        "purchase": "expense",
    }
    out["type"] = out["type"].map(type_mapping).fillna(out["type"])
    out = out[out["type"].isin(["income", "expense"])]

    session: Session = SessionLocal()
    try:
        for _, row in out.iterrows():
            txn = Transaction(
                date=row["date"].date(),
                type=row["type"],
                category=str(row.get("category", "")) if not pd.isna(row.get("category", "")) else "",
                description=str(row.get("description", "")) if not pd.isna(row.get("description", "")) else "",
                amount=float(row["amount"]),
                customer=str(row.get("customer", "")) if not pd.isna(row.get("customer", "")) else "",
                product=str(row.get("product", "")) if not pd.isna(row.get("product", "")) else "",
            )
            session.add(txn)
        session.commit()
    finally:
        session.close()


# ---------------------------
# Analytics: Stats for dashboard
# ---------------------------

def get_dashboard_stats():
    session: Session = SessionLocal()
    try:
        txns = session.query(Transaction).all()
        total_income = sum(t.amount for t in txns if t.type == "income")
        total_expense = sum(t.amount for t in txns if t.type == "expense")
        net_profit = total_income - total_expense

        # latest 10 transactions for table
        latest_txns = (
            session.query(Transaction)
            .order_by(Transaction.date.desc(), Transaction.id.desc())
            .limit(10)
            .all()
        )

        # category-wise + monthly summaries
        category_labels = []
        category_values = []
        month_labels = []
        month_values = []

        if txns:
            data = [
                {
                    "date": t.date,
                    "type": t.type,
                    "category": t.category or "Uncategorized",
                    "amount": t.amount,
                }
                for t in txns
            ]
            df = pd.DataFrame(data)

            # Category wise SUM
            cat_sum = (
                df.groupby("category")["amount"]
                .sum()
                .sort_values(ascending=False)
            )
            category_labels = list(cat_sum.index)
            category_values = [float(x) for x in cat_sum.values]

            # Month wise SUM
            df["year_month"] = df["date"].astype("datetime64[ns]").dt.to_period("M").astype(str)
            month_sum = (
                df.groupby("year_month")["amount"]
                .sum()
                .sort_index()
            )
            month_labels = list(month_sum.index)
            month_values = [float(x) for x in month_sum.values]

        return {
            "total_income": total_income,
            "total_expense": total_expense,
            "net_profit": net_profit,
            "latest_txns": latest_txns,
            "category_labels": category_labels,
            "category_values": category_values,
            "month_labels": month_labels,
            "month_values": month_values,
        }
    finally:
        session.close()


# ---------------------------
# ML: Revenue forecast (next 30 days)
# ---------------------------

def revenue_forecast():
    session = SessionLocal()
    try:
        data = session.query(Transaction).filter(Transaction.type == "income").all()
        if len(data) < 5:
            return None, "Not enough income data to build forecast."

        df = pd.DataFrame([{"date": d.date, "amount": d.amount} for d in data])
        df = df.groupby("date")["amount"].sum().reset_index()

        df = df.sort_values("date")
        df["day_index"] = np.arange(len(df))

        # Train simple linear regression model
        X = df[["day_index"]]
        y = df["amount"]
        model = LinearRegression()
        model.fit(X, y)

        # Predict next 30 days
        future_index = np.arange(len(df), len(df) + 30)
        future_X = future_index.reshape(-1, 1)
        preds = model.predict(future_X)

        result = [
            {"day": int(i + 1), "predicted": float(max(p, 0))}
            for i, p in enumerate(preds)
        ]

        total_prediction = sum(p["predicted"] for p in result)

        return total_prediction, result

    finally:
        session.close()


# ---------------------------
# AI helper: summary + date range + GPT
# ---------------------------

def build_business_summary_text():
    """
    Build overall business summary: income, expense, profit, top categories, monthly trend.
    """
    stats = get_dashboard_stats()

    lines = []
    lines.append(f"Total income so far: ₹{stats['total_income']:.2f}")
    lines.append(f"Total expense so far: ₹{stats['total_expense']:.2f}")
    lines.append(f"Net profit so far: ₹{stats['net_profit']:.2f}")

    # Top categories
    cat_labels = stats.get("category_labels", []) or []
    cat_values = stats.get("category_values", []) or []
    if cat_labels:
        top_parts = []
        for label, val in zip(cat_labels[:5], cat_values[:5]):
            top_parts.append(f"{label}: ₹{val:.2f}")
        lines.append("Top categories by spend/income: " + "; ".join(top_parts))

    # Monthly trend
    month_labels = stats.get("month_labels", []) or []
    month_values = stats.get("month_values", []) or []
    if month_labels:
        month_parts = []
        for label, val in zip(month_labels[-6:], month_values[-6:]):
            month_parts.append(f"{label}: ₹{val:.2f}")
        lines.append("Recent monthly totals: " + "; ".join(month_parts))

    return "\n".join(lines)


def parse_date_range(question: str):
    """
    Very simple heuristic-based date range detection from user question.
    Supports: 'last 7 days', 'last 30 days', 'last month'
    """
    q = question.lower()
    today = datetime.today().date()

    if "last 7 days" in q:
        return today - timedelta(days=7), today

    if "last 30 days" in q:
        return today - timedelta(days=30), today

    if "last month" in q:
        first = today.replace(day=1)
        prev_last = first - timedelta(days=1)
        start = prev_last.replace(day=1)
        end = prev_last
        return start, end

    return None, None


def ask_ai(question: str) -> str:
    """
    GPT ko business summary + (optional) date-filtered summary + user question dete hain,
    aur reply text return karte hain.
    """
    if client is None:
        return "AI assistant is not configured (missing OPENAI_API_KEY)."

    business_summary = build_business_summary_text()

    # Date range based filtered summary (SQL level)
    start_date, end_date = parse_date_range(question)
    filtered_summary = ""

    if start_date and end_date:
        session = SessionLocal()
        try:
            qset = (
                session.query(Transaction)
                .filter(Transaction.date >= start_date, Transaction.date <= end_date)
                .all()
            )

            if not qset:
                filtered_summary = f"No transactions found between {start_date} and {end_date}."
            else:
                income = sum(t.amount for t in qset if t.type == "income")
                expense = sum(t.amount for t in qset if t.type == "expense")
                profit = income - expense

                filtered_summary = f"""
For selected period ({start_date} to {end_date}):
Income: ₹{income:.2f}
Expense: ₹{expense:.2f}
Profit: ₹{profit:.2f}
"""
        finally:
            session.close()

    prompt = f"""
You are an AI financial assistant for a small business/freelancer.

Here is the overall business summary:
---
{business_summary}
---

Filtered date summary (if any):
---
{filtered_summary}
---

The user is asking this question about their business data:
"{question}"

Answer in:
- Simple English
- Use Indian Rupee symbol (₹) where relevant
- Include short bullet points if helpful
- Be practical and specific based on the numbers
If data is missing for something, clearly say it.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or "gpt-4.1" / "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful financial analytics assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    return content


# ---------------------------
# Route
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def dashboard():
    chat_answer = None
    chat_question = None

    if request.method == "POST":
        form_type = request.form.get("form_type", "upload")

        # ---------- 1) File upload form ----------
        if form_type == "upload":
            file = request.files.get("file")
            if not file or file.filename == "":
                flash("Please upload a CSV or Excel file.", "error")
                return redirect(url_for("dashboard"))

            try:
                df = read_file_to_df(file)
                save_transactions_from_df(df)
                flash("Data uploaded and saved successfully!", "success")
            except Exception as e:
                flash(f"Error processing file: {e}", "error")

            return redirect(url_for("dashboard"))

        # ---------- 2) AI chat form ----------
        elif form_type == "chat":
            question = request.form.get("question", "").strip()
            chat_question = question
            if not question:
                flash("Please enter a question for the AI assistant.", "error")
            else:
                try:
                    chat_answer = ask_ai(question)
                except Exception as e:
                    chat_answer = None
                    flash(f"AI error: {e}", "error")

    stats = get_dashboard_stats()
    forecast_total, forecast_data = revenue_forecast()

    return render_template(
        "dashboard.html",
        stats=stats,
        chat_answer=chat_answer,
        chat_question=chat_question,
        forecast_total=forecast_total,
        forecast_data=forecast_data,
    )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)
