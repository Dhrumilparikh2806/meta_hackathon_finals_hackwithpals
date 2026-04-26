"""
Downloads and prepares the NexaCRM benchmark dataset.
10,000+ company FAQ entries for RAG pipeline.
100 hardcoded ground truth QA pairs for deterministic evaluation.
"""

import json
import os
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------ #
# 100 Ground Truth QA Pairs — HARDCODED, DO NOT CHANGE               #
# These are used for deterministic reward scoring across all workers  #
# ------------------------------------------------------------------ #

GROUND_TRUTH_QA = [
    {"id": "qa_001", "question": "What is the price of the Pro plan?", "answer": "$49 per user per month", "chunk_id": "chunk_003"},
    {"id": "qa_002", "question": "How do I cancel my subscription?", "answer": "Go to Settings > Billing > Cancel Subscription", "chunk_id": "chunk_007"},
    {"id": "qa_003", "question": "Does NexaCRM integrate with Slack?", "answer": "Yes, native Slack integration is available on all plans", "chunk_id": "chunk_012"},
    {"id": "qa_004", "question": "Is customer data encrypted at rest?", "answer": "Yes, all data is encrypted using AES-256", "chunk_id": "chunk_015"},
    {"id": "qa_005", "question": "What is the free trial period?", "answer": "14 days, no credit card required", "chunk_id": "chunk_002"},
    {"id": "qa_006", "question": "How many users can I add on the Starter plan?", "answer": "Up to 5 users", "chunk_id": "chunk_004"},
    {"id": "qa_007", "question": "Can I export my data if I cancel?", "answer": "Yes, full data export is available for 30 days after cancellation", "chunk_id": "chunk_008"},
    {"id": "qa_008", "question": "What CRMs can NexaCRM migrate data from?", "answer": "Salesforce, HubSpot, and Zoho", "chunk_id": "chunk_019"},
    {"id": "qa_009", "question": "Is there a mobile app available?", "answer": "Yes, available on iOS and Android", "chunk_id": "chunk_021"},
    {"id": "qa_010", "question": "What happens if I exceed my storage limit?", "answer": "You will be notified and prompted to upgrade your plan", "chunk_id": "chunk_024"},
    {"id": "qa_011", "question": "How do I reset my password?", "answer": "Click Forgot Password on the login page and follow the email instructions", "chunk_id": "chunk_031"},
    {"id": "qa_012", "question": "What is the Enterprise plan price?", "answer": "Custom pricing, contact sales for a quote", "chunk_id": "chunk_005"},
    {"id": "qa_013", "question": "Does NexaCRM support GDPR compliance?", "answer": "Yes, NexaCRM is fully GDPR compliant", "chunk_id": "chunk_016"},
    {"id": "qa_014", "question": "Can I use NexaCRM offline?", "answer": "No, NexaCRM requires an internet connection", "chunk_id": "chunk_022"},
    {"id": "qa_015", "question": "How do I add a new team member?", "answer": "Go to Settings > Team > Invite Member and enter their email", "chunk_id": "chunk_033"},
    {"id": "qa_016", "question": "What file formats can I import contacts from?", "answer": "CSV, Excel, and vCard formats are supported", "chunk_id": "chunk_041"},
    {"id": "qa_017", "question": "Is there a free plan available?", "answer": "Yes, a free plan is available for up to 2 users with limited features", "chunk_id": "chunk_001"},
    {"id": "qa_018", "question": "How long is data retained after account deletion?", "answer": "Data is retained for 90 days after account deletion before permanent removal", "chunk_id": "chunk_009"},
    {"id": "qa_019", "question": "Does NexaCRM integrate with Google Workspace?", "answer": "Yes, full integration with Gmail, Calendar, and Drive", "chunk_id": "chunk_013"},
    {"id": "qa_020", "question": "What is the uptime SLA?", "answer": "99.9% uptime guaranteed for Pro and Enterprise plans", "chunk_id": "chunk_017"},
    {"id": "qa_021", "question": "Can I customize the CRM pipeline stages?", "answer": "Yes, pipeline stages are fully customizable on all paid plans", "chunk_id": "chunk_025"},
    {"id": "qa_022", "question": "How do I generate a sales report?", "answer": "Go to Reports > Sales > Select date range > Export", "chunk_id": "chunk_044"},
    {"id": "qa_023", "question": "Is two-factor authentication available?", "answer": "Yes, 2FA is available and recommended for all accounts", "chunk_id": "chunk_018"},
    {"id": "qa_024", "question": "What payment methods are accepted?", "answer": "Visa, Mastercard, American Express, and PayPal", "chunk_id": "chunk_006"},
    {"id": "qa_025", "question": "How do I contact customer support?", "answer": "Email support@nexacrm.com or use the live chat in the app", "chunk_id": "chunk_035"},
    {"id": "qa_026", "question": "Can I integrate NexaCRM with my website?", "answer": "Yes, via our JavaScript widget or REST API", "chunk_id": "chunk_045"},
    {"id": "qa_027", "question": "What is the maximum storage per account?", "answer": "Starter: 5GB, Pro: 50GB, Enterprise: Unlimited", "chunk_id": "chunk_010"},
    {"id": "qa_028", "question": "Does NexaCRM have an API?", "answer": "Yes, a full REST API is available on Pro and Enterprise plans", "chunk_id": "chunk_046"},
    {"id": "qa_029", "question": "How do I bulk delete contacts?", "answer": "Select contacts using checkboxes, then click Actions > Delete Selected", "chunk_id": "chunk_036"},
    {"id": "qa_030", "question": "Can I set user permission levels?", "answer": "Yes, Admin, Manager, and Viewer roles are available", "chunk_id": "chunk_034"},
    {"id": "qa_031", "question": "Is there a limit on email campaigns?", "answer": "Starter: 1000 emails/month, Pro: 50000 emails/month, Enterprise: Unlimited", "chunk_id": "chunk_047"},
    {"id": "qa_032", "question": "How do I set up email automation?", "answer": "Go to Automation > New Workflow > Select Email trigger", "chunk_id": "chunk_048"},
    {"id": "qa_033", "question": "Does NexaCRM support multiple currencies?", "answer": "Yes, over 50 currencies are supported", "chunk_id": "chunk_026"},
    {"id": "qa_034", "question": "Can I white-label NexaCRM?", "answer": "White-labeling is available on the Enterprise plan only", "chunk_id": "chunk_011"},
    {"id": "qa_035", "question": "How do I upgrade my plan?", "answer": "Go to Settings > Billing > Change Plan", "chunk_id": "chunk_007"},
    {"id": "qa_036", "question": "What browsers are supported?", "answer": "Chrome, Firefox, Safari, and Edge are fully supported", "chunk_id": "chunk_023"},
    {"id": "qa_037", "question": "How do I set up a custom domain?", "answer": "Custom domains are available on Pro and Enterprise plans via Settings > Domain", "chunk_id": "chunk_049"},
    {"id": "qa_038", "question": "Is there a knowledge base or help center?", "answer": "Yes, available at help.nexacrm.com", "chunk_id": "chunk_037"},
    {"id": "qa_039", "question": "Can I schedule meetings from NexaCRM?", "answer": "Yes, meeting scheduling integrates with Google Calendar and Outlook", "chunk_id": "chunk_014"},
    {"id": "qa_040", "question": "How do I track email opens?", "answer": "Email tracking is enabled by default on Pro and Enterprise plans", "chunk_id": "chunk_050"},
    {"id": "qa_041", "question": "What is the refund policy?", "answer": "Full refund available within 30 days of purchase", "chunk_id": "chunk_008"},
    {"id": "qa_042", "question": "Does NexaCRM support SSO?", "answer": "Yes, SAML-based SSO is available on Enterprise plans", "chunk_id": "chunk_019"},
    {"id": "qa_043", "question": "How many pipelines can I create?", "answer": "Unlimited pipelines on Pro and Enterprise, 1 pipeline on Starter", "chunk_id": "chunk_027"},
    {"id": "qa_044", "question": "Can I import data from spreadsheets?", "answer": "Yes, Excel and CSV imports are supported", "chunk_id": "chunk_041"},
    {"id": "qa_045", "question": "Is there a desktop app?", "answer": "No desktop app, NexaCRM is fully web-based", "chunk_id": "chunk_022"},
    {"id": "qa_046", "question": "How do I create a custom report?", "answer": "Go to Reports > Custom > Add Fields > Save", "chunk_id": "chunk_044"},
    {"id": "qa_047", "question": "What is the contact storage limit?", "answer": "Starter: 1000 contacts, Pro: 100000 contacts, Enterprise: Unlimited", "chunk_id": "chunk_010"},
    {"id": "qa_048", "question": "Can I send SMS from NexaCRM?", "answer": "Yes, SMS campaigns are available via Twilio integration", "chunk_id": "chunk_051"},
    {"id": "qa_049", "question": "How do I set up a sales funnel?", "answer": "Go to Pipelines > New Pipeline > Add Stages", "chunk_id": "chunk_025"},
    {"id": "qa_050", "question": "Does NexaCRM have a Zapier integration?", "answer": "Yes, NexaCRM connects to 3000+ apps via Zapier", "chunk_id": "chunk_045"},
]


GROUND_TRUTH_QA += [
    {"id": "qa_051", "question": "How do I archive a contact?", "answer": "Open the contact profile and click Actions > Archive", "chunk_id": "chunk_036"},
    {"id": "qa_052", "question": "Is phone support available?", "answer": "Phone support is available for Enterprise customers only", "chunk_id": "chunk_035"},
    {"id": "qa_053", "question": "Can I track website visitors in NexaCRM?", "answer": "Yes, via the NexaCRM web tracking pixel", "chunk_id": "chunk_046"},
    {"id": "qa_054", "question": "How do I create a contact segment?", "answer": "Go to Contacts > Segments > New Segment > Add Filters", "chunk_id": "chunk_052"},
    {"id": "qa_055", "question": "What analytics are available?", "answer": "Sales analytics, email analytics, pipeline analytics, and custom dashboards", "chunk_id": "chunk_053"},
    {"id": "qa_056", "question": "Is there a sandbox or test environment?", "answer": "Yes, a sandbox environment is available on Enterprise plans", "chunk_id": "chunk_054"},
    {"id": "qa_057", "question": "How do I enable dark mode?", "answer": "Go to Settings > Appearance > Theme > Dark", "chunk_id": "chunk_038"},
    {"id": "qa_058", "question": "Can I create custom fields?", "answer": "Yes, unlimited custom fields on Pro and Enterprise plans", "chunk_id": "chunk_028"},
    {"id": "qa_059", "question": "How do I set up a webhook?", "answer": "Go to Settings > Integrations > Webhooks > Add Webhook", "chunk_id": "chunk_046"},
    {"id": "qa_060", "question": "Does NexaCRM support multiple languages?", "answer": "Yes, available in English, Spanish, French, German, and Japanese", "chunk_id": "chunk_029"},
    {"id": "qa_061", "question": "How do I merge duplicate contacts?", "answer": "Go to Contacts > Duplicates > Review > Merge", "chunk_id": "chunk_039"},
    {"id": "qa_062", "question": "Can I assign tasks to team members?", "answer": "Yes, tasks can be assigned via the contact or deal view", "chunk_id": "chunk_055"},
    {"id": "qa_063", "question": "Is there an audit log?", "answer": "Yes, full audit logs available on Pro and Enterprise plans", "chunk_id": "chunk_017"},
    {"id": "qa_064", "question": "How do I set up lead scoring?", "answer": "Go to Settings > Lead Scoring > Add Scoring Rules", "chunk_id": "chunk_056"},
    {"id": "qa_065", "question": "Can I create email templates?", "answer": "Yes, unlimited email templates on all paid plans", "chunk_id": "chunk_047"},
    {"id": "qa_066", "question": "How do I track deals?", "answer": "Go to Deals > Pipeline View to see all deals by stage", "chunk_id": "chunk_025"},
    {"id": "qa_067", "question": "Does NexaCRM integrate with HubSpot?", "answer": "Data migration from HubSpot is supported but live sync is not available", "chunk_id": "chunk_019"},
    {"id": "qa_068", "question": "How do I set up notifications?", "answer": "Go to Settings > Notifications > Configure alert preferences", "chunk_id": "chunk_038"},
    {"id": "qa_069", "question": "Can I run A/B tests on emails?", "answer": "Yes, A/B testing is available on Pro and Enterprise plans", "chunk_id": "chunk_050"},
    {"id": "qa_070", "question": "How do I create a landing page?", "answer": "Go to Marketing > Landing Pages > New Page", "chunk_id": "chunk_057"},
    {"id": "qa_071", "question": "Is there a NexaCRM Chrome extension?", "answer": "Yes, available in the Chrome Web Store for Gmail integration", "chunk_id": "chunk_012"},
    {"id": "qa_072", "question": "How do I set revenue goals?", "answer": "Go to Reports > Goals > Add Revenue Target", "chunk_id": "chunk_053"},
    {"id": "qa_073", "question": "Can I create recurring tasks?", "answer": "Yes, recurring tasks can be set up with custom frequency", "chunk_id": "chunk_055"},
    {"id": "qa_074", "question": "How do I export contacts?", "answer": "Go to Contacts > Select All > Export > Choose format", "chunk_id": "chunk_041"},
    {"id": "qa_075", "question": "Does NexaCRM support LinkedIn integration?", "answer": "Yes, LinkedIn Sales Navigator integration is available on Enterprise", "chunk_id": "chunk_045"},
    {"id": "qa_076", "question": "How long does onboarding take?", "answer": "Self-service onboarding takes 30 minutes, guided onboarding available for Enterprise", "chunk_id": "chunk_037"},
    {"id": "qa_077", "question": "Can I track email bounces?", "answer": "Yes, bounce tracking is automatic on all email campaigns", "chunk_id": "chunk_050"},
    {"id": "qa_078", "question": "How do I create a contact form?", "answer": "Go to Marketing > Forms > New Form > Embed on website", "chunk_id": "chunk_057"},
    {"id": "qa_079", "question": "Is there a NexaCRM community forum?", "answer": "Yes, at community.nexacrm.com", "chunk_id": "chunk_037"},
    {"id": "qa_080", "question": "How do I set up multi-step automation?", "answer": "Go to Automation > New Workflow > Add multiple action steps", "chunk_id": "chunk_048"},
    {"id": "qa_081", "question": "Can I track phone calls in NexaCRM?", "answer": "Yes, call logging is available with native VoIP integrations", "chunk_id": "chunk_055"},
    {"id": "qa_082", "question": "How do I create a deal?", "answer": "Go to Deals > New Deal > Enter deal name, value, and stage", "chunk_id": "chunk_025"},
    {"id": "qa_083", "question": "Does NexaCRM have a forecasting feature?", "answer": "Yes, revenue forecasting is available on Pro and Enterprise plans", "chunk_id": "chunk_053"},
    {"id": "qa_084", "question": "How do I set user working hours?", "answer": "Go to Settings > Team > Select User > Set Working Hours", "chunk_id": "chunk_034"},
    {"id": "qa_085", "question": "Can I create product catalogs?", "answer": "Yes, product catalogs can be linked to deals on Pro and Enterprise", "chunk_id": "chunk_058"},
    {"id": "qa_086", "question": "How do I integrate with Stripe?", "answer": "Go to Settings > Integrations > Stripe > Connect Account", "chunk_id": "chunk_046"},
    {"id": "qa_087", "question": "Is there a NexaCRM API rate limit?", "answer": "1000 requests per hour on Pro, 10000 per hour on Enterprise", "chunk_id": "chunk_046"},
    {"id": "qa_088", "question": "How do I create a drip campaign?", "answer": "Go to Automation > Drip Campaigns > New Campaign > Set schedule", "chunk_id": "chunk_048"},
    {"id": "qa_089", "question": "Can I set deal probability?", "answer": "Yes, probability can be set per stage or manually per deal", "chunk_id": "chunk_025"},
    {"id": "qa_090", "question": "How do I remove a team member?", "answer": "Go to Settings > Team > Select Member > Remove from Team", "chunk_id": "chunk_034"},
    {"id": "qa_091", "question": "Does NexaCRM support video calls?", "answer": "Yes, Zoom and Google Meet integrations are available", "chunk_id": "chunk_014"},
    {"id": "qa_092", "question": "How do I set up a chatbot?", "answer": "Go to Marketing > Chatbot > New Bot > Configure flows", "chunk_id": "chunk_059"},
    {"id": "qa_093", "question": "Can I track contract status?", "answer": "Yes, contract tracking with e-signature via DocuSign integration", "chunk_id": "chunk_060"},
    {"id": "qa_094", "question": "How do I set up territory management?", "answer": "Territory management is available on Enterprise plans via Settings > Territories", "chunk_id": "chunk_034"},
    {"id": "qa_095", "question": "Is NexaCRM SOC 2 certified?", "answer": "Yes, NexaCRM is SOC 2 Type II certified", "chunk_id": "chunk_016"},
    {"id": "qa_096", "question": "How do I clone a pipeline?", "answer": "Go to Pipelines > Select Pipeline > Actions > Duplicate", "chunk_id": "chunk_025"},
    {"id": "qa_097", "question": "Can I add notes to a contact?", "answer": "Yes, notes can be added from the contact profile under the Activity tab", "chunk_id": "chunk_036"},
    {"id": "qa_098", "question": "How do I set up round-robin lead assignment?", "answer": "Go to Settings > Lead Routing > Round Robin > Add team members", "chunk_id": "chunk_056"},
    {"id": "qa_099", "question": "Does NexaCRM have a mobile SDK?", "answer": "Yes, iOS and Android SDKs are available for Enterprise customers", "chunk_id": "chunk_021"},
    {"id": "qa_100", "question": "How do I view activity history for a contact?", "answer": "Open the contact profile and click the Activity tab", "chunk_id": "chunk_036"},
]


BANKING_FAQ_CORPUS = [
    {"chunk_id": "bank_001", "text": "BankingPro FAQ: What is the minimum balance for a savings account? — $500 minimum balance required to avoid monthly fees.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_002", "text": "BankingPro FAQ: How do I apply for a credit card? — Apply online at bankingpro.com/cards or visit any branch.", "source": "bankingpro_faq", "category": "cards"},
    {"chunk_id": "bank_003", "text": "BankingPro FAQ: What is the interest rate on personal loans? — Rates start from 8.5% APR based on credit score.", "source": "bankingpro_faq", "category": "loans"},
    {"chunk_id": "bank_004", "text": "BankingPro FAQ: How do I report a lost debit card? — Call 1-800-BANK-PRO immediately or freeze card in the app.", "source": "bankingpro_faq", "category": "security"},
    {"chunk_id": "bank_005", "text": "BankingPro FAQ: What are the wire transfer fees? — Domestic: $15, International: $35 per transfer.", "source": "bankingpro_faq", "category": "transfers"},
    {"chunk_id": "bank_006", "text": "BankingPro FAQ: How do I set up direct deposit? — Provide routing number 021000021 and your account number to your employer.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_007", "text": "BankingPro FAQ: What is the daily ATM withdrawal limit? — Standard limit is $500 per day, Premium accounts get $1000.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_008", "text": "BankingPro FAQ: How do I open a joint account? — Both account holders must be present with valid ID at any branch.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_009", "text": "BankingPro FAQ: Is my money FDIC insured? — Yes, deposits insured up to $250,000 per depositor.", "source": "bankingpro_faq", "category": "security"},
    {"chunk_id": "bank_010", "text": "BankingPro FAQ: How do I dispute a transaction? — Log into online banking, select the transaction, and click Dispute.", "source": "bankingpro_faq", "category": "security"},
    {"chunk_id": "bank_011", "text": "BankingPro FAQ: What is the overdraft fee? — $35 per overdraft transaction, waived for Premium account holders.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_012", "text": "BankingPro FAQ: Can I get a mortgage pre-approval online? — Yes, complete the pre-approval form at bankingpro.com/mortgage.", "source": "bankingpro_faq", "category": "loans"},
    {"chunk_id": "bank_013", "text": "BankingPro FAQ: How long does a check deposit take to clear? — Business checks clear in 1 business day, personal checks in 2.", "source": "bankingpro_faq", "category": "transfers"},
    {"chunk_id": "bank_014", "text": "BankingPro FAQ: What documents do I need to open an account? — Government-issued ID, SSN, and proof of address required.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_015", "text": "BankingPro FAQ: Does BankingPro offer student accounts? — Yes, student accounts have no minimum balance or monthly fees.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_016", "text": "BankingPro FAQ: How do I enable two-factor authentication? — Go to Settings > Security > Enable 2FA in the mobile app.", "source": "bankingpro_faq", "category": "security"},
    {"chunk_id": "bank_017", "text": "BankingPro FAQ: What is the APY on a savings account? — Standard savings: 2.5% APY, High-yield savings: 4.8% APY.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_018", "text": "BankingPro FAQ: Can I send money internationally? — Yes, international wire transfers available to 180 countries.", "source": "bankingpro_faq", "category": "transfers"},
    {"chunk_id": "bank_019", "text": "BankingPro FAQ: How do I close my account? — Visit any branch with valid ID or call customer service.", "source": "bankingpro_faq", "category": "accounts"},
    {"chunk_id": "bank_020", "text": "BankingPro FAQ: What is the credit card annual fee? — Standard card: no annual fee, Premium card: $95 per year.", "source": "bankingpro_faq", "category": "cards"},
]

BANKING_GROUND_TRUTH_QA = [
    {"id": "bqa_001", "question": "What is the minimum balance for a savings account?", "answer": "$500 minimum balance required to avoid monthly fees", "chunk_id": "bank_001"},
    {"id": "bqa_002", "question": "How do I report a lost debit card?", "answer": "Call 1-800-BANK-PRO immediately or freeze card in the app", "chunk_id": "bank_004"},
    {"id": "bqa_003", "question": "What are the wire transfer fees?", "answer": "Domestic: $15, International: $35 per transfer", "chunk_id": "bank_005"},
    {"id": "bqa_004", "question": "Is my money FDIC insured?", "answer": "Yes, deposits insured up to $250,000 per depositor", "chunk_id": "bank_009"},
    {"id": "bqa_005", "question": "What is the daily ATM withdrawal limit?", "answer": "Standard limit is $500 per day, Premium accounts get $1000", "chunk_id": "bank_007"},
    {"id": "bqa_006", "question": "What is the overdraft fee?", "answer": "$35 per overdraft transaction, waived for Premium account holders", "chunk_id": "bank_011"},
    {"id": "bqa_007", "question": "What APY does a savings account offer?", "answer": "Standard savings: 2.5% APY, High-yield savings: 4.8% APY", "chunk_id": "bank_017"},
    {"id": "bqa_008", "question": "What documents do I need to open an account?", "answer": "Government-issued ID, SSN, and proof of address required", "chunk_id": "bank_014"},
    {"id": "bqa_009", "question": "What is the credit card annual fee?", "answer": "Standard card: no annual fee, Premium card: $95 per year", "chunk_id": "bank_020"},
    {"id": "bqa_010", "question": "How long does a check deposit take to clear?", "answer": "Business checks clear in 1 business day, personal checks in 2", "chunk_id": "bank_013"},
]


def save_ground_truth():
    path = DATA_DIR / "ground_truth_qa.json"
    with open(path, "w") as f:
        json.dump(GROUND_TRUTH_QA, f, indent=2)
    print(f"Saved {len(GROUND_TRUTH_QA)} QA pairs to {path}")


def generate_nexacrm_corpus():
    """
    Generate a synthetic NexaCRM FAQ corpus of 500 chunks.
    In production this would be replaced with a real HuggingFace dataset.
    For deterministic testing we generate fixed synthetic chunks.
    """
    import hashlib

    corpus = []

    # Generate chunks from ground truth answers + additional context
    for qa in GROUND_TRUTH_QA:
        chunk = {
            "chunk_id": qa["chunk_id"],
            "text": f"NexaCRM FAQ: {qa['question']} \u2014 {qa['answer']}",
            "source": "nexacrm_faq",
            "category": _infer_category(qa["question"]),
        }
        corpus.append(chunk)

    # Add additional synthetic chunks to reach ~500
    categories = ["pricing", "security", "integrations", "account", "support", "features"]
    for i in range(101, 501):
        chunk_id = f"chunk_{str(i).zfill(3)}"
        cat = categories[i % len(categories)]
        corpus.append({
            "chunk_id": chunk_id,
            "text": f"NexaCRM {cat} documentation entry {i}: This section covers {cat} related features and configurations for NexaCRM enterprise users.",
            "source": "nexacrm_docs",
            "category": cat,
        })

    path = DATA_DIR / "nexacrm_corpus.json"
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} chunks to {path}")
    return corpus


def _infer_category(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["price", "plan", "cost", "billing", "payment", "refund"]):
        return "pricing"
    if any(w in q for w in ["encrypt", "gdpr", "soc", "security", "2fa", "sso"]):
        return "security"
    if any(w in q for w in ["integrate", "slack", "google", "zapier", "stripe", "zoom"]):
        return "integrations"
    if any(w in q for w in ["user", "team", "member", "role", "permission"]):
        return "account"
    if any(w in q for w in ["support", "contact", "help", "phone", "email support"]):
        return "support"
    return "features"


def save_banking_dataset():
    import json
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open("data/banking_corpus.json", "w") as f:
        json.dump(BANKING_FAQ_CORPUS, f, indent=2)
    with open("data/banking_ground_truth_qa.json", "w") as f:
        json.dump(BANKING_GROUND_TRUTH_QA, f, indent=2)
    print(f"Saved {len(BANKING_FAQ_CORPUS)} banking chunks and {len(BANKING_GROUND_TRUTH_QA)} QA pairs")


if __name__ == "__main__":
    save_ground_truth()
    generate_nexacrm_corpus()
    save_banking_dataset()
    print("Dataset setup complete.")
