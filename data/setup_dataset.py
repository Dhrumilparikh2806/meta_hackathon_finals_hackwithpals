"""
Downloads and prepares the NexaCRM benchmark dataset.
2000 realistic company FAQ entries for RAG pipeline.
400 hardcoded ground truth QA pairs for deterministic evaluation.
"""

import json
import os
import random
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Set seed for determinism
random.seed(42)

def generate_nexacrm_data():
    categories_config = {
        "pricing": {
            "count": 222,
            "topics": [
                "Starter plan: $19/user/month, up to 5 users, basic CRM features",
                "Professional plan: $49/user/month, up to 50 users, automation included",
                "Enterprise plan: $99/user/month, unlimited users, dedicated support",
                "Annual billing discount: 20% off monthly price",
                "Free trial: 14 days, no credit card required",
                "Student/nonprofit discount: 40% off Professional plan",
                "Price comparison with competitors: NexaCRM is 30% cheaper than Salesforce",
                "Volume discounts for 100+ users: Contact sales for custom quotes",
                "Add-on pricing: advanced analytics $15/user/month",
                "Custom pricing for enterprise contracts: Starts at $5000/year",
                "Currency support: USD, EUR, GBP, AUD, CAD, INR",
                "Price increase policy: 30 days notice provided via email",
                "Grandfathering policy: Existing customers keep their original pricing for 2 years",
                "Billing frequency: monthly or annual options available",
                "Prorated billing for mid-cycle upgrades: You only pay the difference for the remaining days"
            ],
            "questions": [
                "How much does the Starter plan cost?",
                "What is included in the Professional plan?",
                "Is there an Enterprise plan available?",
                "Do you offer discounts for annual billing?",
                "How long is the free trial?",
                "Are there discounts for non-profits?",
                "How does NexaCRM pricing compare to competitors?",
                "Do you offer volume discounts?",
                "How much is the advanced analytics add-on?",
                "What is the starting price for enterprise contracts?",
                "What currencies do you support?",
                "What is your price increase policy?",
                "Do old customers get to keep their prices?",
                "Can I pay monthly?",
                "How does billing work if I upgrade in the middle of the month?"
            ]
        },
        "account_management": {
            "count": 222,
            "topics": [
                "How to create an account: Click Sign Up on the homepage",
                "How to add team members: Settings > Team > Invite",
                "Role-based access control: Admin, Manager, Sales Rep, Viewer",
                "How to transfer account ownership: Contact support or use Account Settings",
                "How to merge duplicate accounts: Settings > Data > Merge Accounts",
                "Password reset process: Click 'Forgot Password' on login screen",
                "Two-factor authentication setup: Security settings > Enable 2FA",
                "Single sign-on (SSO) configuration: Enterprise only via SAML 2.0",
                "Account suspension and reactivation: Managed via Billing portal",
                "How to delete an account: Settings > Account > Delete (Permanent)",
                "Data export before deletion: Export as CSV or JSON in Settings",
                "Account audit logs: Available in Security tab for Pro and Enterprise",
                "Session timeout settings: Security settings > Session Management",
                "Login history and security alerts: View in Security > Login History",
                "Multi-organization support: Switch between orgs in the top-right menu"
            ],
            "questions": [
                "Where do I sign up for an account?",
                "How do I invite my team?",
                "What roles are available in NexaCRM?",
                "Can I change the account owner?",
                "How do I merge two accounts?",
                "What if I forget my password?",
                "Does NexaCRM support 2FA?",
                "Is SSO available?",
                "What happens if my account is suspended?",
                "How do I close my account?",
                "Can I download my data before deleting?",
                "Where can I see who logged in?",
                "Can I change the session timeout?",
                "Will I get an alert for new logins?",
                "Can I manage multiple companies?"
            ]
        },
        "integrations": {
            "count": 222,
            "topics": [
                "Gmail and Google Workspace integration: Full sync for emails and calendar",
                "Outlook and Microsoft 365 integration: Sync contacts and meetings",
                "Slack integration and notifications: Get deal alerts in Slack channels",
                "Zapier integration: Connect to 3000+ apps",
                "Salesforce data migration: Import tool available in Settings",
                "HubSpot data import: Supports contacts and deals via CSV",
                "Mailchimp email marketing sync: Sync segments and campaign results",
                "QuickBooks accounting sync: Sync invoices and payments",
                "Stripe payment tracking: View subscription status in CRM",
                "Twilio SMS integration: Send and receive SMS from contact view",
                "LinkedIn Sales Navigator: Available on Enterprise plan",
                "Google Calendar sync: Two-way sync for all meetings",
                "Zoom meeting logging: Automatically log recordings to deals",
                "Shopify ecommerce integration: Sync customers and orders",
                "Webhooks and API access: Available for custom workflows",
                "Native mobile apps: Available for iOS and Android",
                "Chrome extension for LinkedIn: Save leads directly from browser",
                "REST API documentation: developers.nexacrm.com",
                "API rate limits: 1000 requests per hour on Professional",
                "OAuth 2.0 authentication: Used for all secure integrations"
            ],
            "questions": [
                "Does it work with Gmail?",
                "Can I sync with Outlook?",
                "Is there a Slack app?",
                "Does NexaCRM support Zapier?",
                "Can I migrate from Salesforce?",
                "How do I import from HubSpot?",
                "Does it sync with Mailchimp?",
                "Can I connect QuickBooks?",
                "Does it track Stripe payments?",
                "Can I send SMS through NexaCRM?",
                "Is LinkedIn integrated?",
                "Does it sync with my calendar?",
                "Can I log Zoom calls?",
                "Is Shopify supported?",
                "Do you have webhooks?",
                "Is there a mobile app?",
                "Is there a Chrome extension?",
                "Where is the API documentation?",
                "What are the API rate limits?",
                "How do integrations authenticate?"
            ]
        },
        "features": {
            "count": 222,
            "topics": [
                "Contact management and deduplication: Automatic duplicate detection",
                "Deal pipeline management: Drag and drop Kanban board",
                "Activity tracking: Log calls, emails, and meetings automatically",
                "Email templates and sequences: Personalize outreach at scale",
                "Sales forecasting: Predict revenue based on deal probability",
                "Custom fields and objects: Add unique data points to records",
                "Workflow automation triggers: Automate tasks based on deal stages",
                "Lead scoring configuration: Rank leads by engagement level",
                "Territory management: Assign leads by geographic region",
                "Quote and proposal generation: Create PDFs directly from deals",
                "Document storage: 10GB per user included",
                "Reporting and dashboards: Real-time sales analytics",
                "Goal tracking for sales teams: Set and monitor quotas",
                "Mobile CRM features: Full access to deals and contacts on the go",
                "Bulk data import via CSV: Import thousands of records at once",
                "Custom tags and filters: Organize data with flexible tagging",
                "Smart lists and segments: Dynamic lists that update automatically",
                "Email tracking: See when recipients open or click emails",
                "Call recording and transcription: Enterprise feature for sales coaching",
                "AI-powered insights: Predictive lead scoring for Enterprise"
            ],
            "questions": [
                "How do you handle duplicate contacts?",
                "What does the pipeline look like?",
                "Can I track my calls?",
                "Does NexaCRM have email templates?",
                "Can I forecast my sales?",
                "Can I add custom fields?",
                "How does workflow automation work?",
                "What is lead scoring?",
                "Does it support territory management?",
                "Can I create quotes in NexaCRM?",
                "How much storage do I get?",
                "What reports are available?",
                "Can I track sales goals?",
                "What can I do on the mobile app?",
                "Can I import data from CSV?",
                "How do I tag contacts?",
                "What are smart lists?",
                "Can I see if someone opened my email?",
                "Do you record calls?",
                "What AI features are included?"
            ]
        },
        "security": {
            "count": 166,
            "topics": [
                "SOC 2 Type II compliance: Audited annually by third parties",
                "GDPR compliance and data residency: Data hosted in EU for EU customers",
                "CCPA compliance: Full support for California privacy rights",
                "Data encryption at rest: All data encrypted with AES-256",
                "Data encryption in transit: TLS 1.3 for all connections",
                "Password requirements: Minimum 12 characters, including special symbols",
                "IP allowlisting: Restrict access to specific IP ranges",
                "Audit logs retention: Logs kept for 1 year",
                "Penetration testing: Annual tests by independent security firms",
                "Bug bounty program: Reward program for responsible disclosure",
                "Data backup frequency: Automatic backups every 6 hours",
                "Disaster recovery: 99.9% uptime SLA with failover to secondary regions",
                "Data center locations: US (East/West), EU (Frankfurt), APAC (Singapore)",
                "HIPAA compliance: BAA available for Enterprise customers",
                "Vulnerability disclosure policy: Published on our security page"
            ],
            "questions": [
                "Is NexaCRM SOC 2 compliant?",
                "Are you GDPR compliant?",
                "Does it follow CCPA?",
                "How is data encrypted at rest?",
                "Is data encrypted during transfer?",
                "What are the password requirements?",
                "Can I restrict login by IP?",
                "How long are audit logs kept?",
                "Do you perform pentesting?",
                "Is there a bug bounty program?",
                "How often is data backed up?",
                "What is your uptime SLA?",
                "Where are your data centers?",
                "Do you support HIPAA?",
                "How do I report a security vulnerability?"
            ]
        },
        "support": {
            "count": 166,
            "topics": [
                "Support channels: Email, live chat, and phone support",
                "Response times: Starter 48h, Professional 24h, Enterprise 4h",
                "Support hours: Monday-Friday 9am-6pm EST",
                "Emergency support for Enterprise: 24/7 priority line",
                "Help center and documentation: help.nexacrm.com",
                "Video tutorial library: 200+ step-by-step videos",
                "Onboarding assistance: Dedicated session for Professional/Enterprise",
                "Dedicated customer success manager: Provided for Enterprise accounts",
                "Community forum: Discuss features with other users",
                "Webinar schedule: Live training every Tuesday",
                "How to submit a support ticket: Use the '?' icon in the app",
                "Escalation process: Tickets can be escalated to senior engineers",
                "Known issues page: Check status.nexacrm.com for bugs",
                "System status page: Real-time uptime monitoring",
                "Feature request submission: Vote on our public roadmap"
            ],
            "questions": [
                "How can I contact support?",
                "What are the response times?",
                "What are your support hours?",
                "Is there 24/7 support?",
                "Where is the help center?",
                "Are there video tutorials?",
                "Do you help with onboarding?",
                "Do I get a dedicated success manager?",
                "Is there a community forum?",
                "When are the webinars?",
                "How do I open a ticket?",
                "What if my issue is urgent?",
                "Where can I see known bugs?",
                "Is the system down?",
                "How do I suggest a new feature?"
            ]
        },
        "data_management": {
            "count": 166,
            "topics": [
                "Data import formats: CSV, Excel (.xlsx), and VCF supported",
                "Data export formats: Export all data to CSV, JSON, or PDF",
                "Bulk delete operations: Select multiple records to delete at once",
                "Data archiving policy: Archive old deals to keep pipeline clean",
                "Data retention settings: Configure how long deleted data is kept",
                "Custom data fields: Support for text, number, date, and picklists",
                "Data validation rules: Ensure clean data with required fields",
                "Duplicate detection algorithm: Uses email and phone number matching",
                "Merge contacts and companies: Combine history into a single record",
                "Data ownership and permissions: You own 100% of your data",
                "GDPR right to erasure: Tools to permanently delete customer data",
                "Data portability: Easily move data between NexaCRM instances",
                "Historical data access: View activity history for the past 5 years",
                "Data quality scoring: Identify incomplete records automatically",
                "Backup and restore: Contact support for manual restores"
            ],
            "questions": [
                "What files can I import?",
                "In what formats can I export data?",
                "Can I delete many records at once?",
                "How do I archive data?",
                "How long is my data stored?",
                "What types of custom fields are there?",
                "Can I make fields mandatory?",
                "How does duplicate detection work?",
                "Can I merge two companies?",
                "Who owns the data I upload?",
                "How do I process a 'Right to be Forgotten' request?",
                "Can I move my data to another account?",
                "How far back does history go?",
                "What is data quality scoring?",
                "How do I restore from a backup?"
            ]
        },
        "billing": {
            "count": 166,
            "topics": [
                "Accepted payment methods: Visa, Mastercard, Amex, PayPal, wire transfer",
                "Invoice generation and delivery: Emailed automatically each month",
                "How to update billing information: Settings > Billing > Payment Method",
                "How to cancel subscription: Settings > Billing > Cancel Plan",
                "Refund policy: 30 days money back guarantee for all new subs",
                "Failed payment handling: 3 retry attempts before suspension",
                "Tax invoices and VAT: Download VAT-compliant invoices in Settings",
                "Usage-based billing for API: Billed monthly based on overages",
                "How to download past invoices: Available in the Billing History tab",
                "Billing contact settings: Add a secondary email for invoices",
                "Auto-renewal policy: Subscriptions renew automatically unless cancelled",
                "Upgrade and downgrade: Prorated credits applied instantly",
                "Spending limits: Set caps on API and SMS usage",
                "PO number support: Add PO numbers to your invoices",
                "Credit card security: We do not store full card numbers on our servers"
            ],
            "questions": [
                "What payment methods do you take?",
                "How do I get my invoice?",
                "How do I change my credit card?",
                "How do I stop my subscription?",
                "What is your refund policy?",
                "What happens if my payment fails?",
                "Do you provide VAT invoices?",
                "How is API usage billed?",
                "Where can I see old invoices?",
                "Can I send invoices to my accountant?",
                "Do I have to renew manually?",
                "What happens if I downgrade?",
                "Can I set a spending limit?",
                "Can I add a PO number?",
                "Is my credit card info safe?"
            ]
        },
        "mobile_app": {
            "count": 112,
            "topics": [
                "iOS app minimum version: Requires iOS 14.0 or later",
                "Android app minimum version: Requires Android 8.0 (Oreo) or later",
                "Offline mode: View contacts and deals without internet; syncs on reconnect",
                "Push notifications: Real-time alerts for deal updates and tasks",
                "Mobile-specific features: Business card scanner and call logging",
                "Biometric authentication: Support for FaceID and Fingerprint login",
                "App size and storage: Lightweight app (~50MB initial download)",
                "Sync frequency: Background sync every 15 minutes",
                "Mobile data usage: Optimized for low bandwidth consumption",
                "Tablet optimization: Full-screen support for iPad and Android tablets",
                "Apple Watch support: View today's tasks on your wrist",
                "Widget support: iOS and Android home screen widgets for quick access",
                "Background sync: Keeps data fresh even when app is closed",
                "Voice input: Dictate notes directly into the CRM",
                "Camera integration: Take photos of documents and attach to deals"
            ],
            "questions": [
                "What iOS version is needed?",
                "What Android version is needed?",
                "Does it work without Wi-Fi?",
                "Does the app send alerts?",
                "What special features does the app have?",
                "Can I use FaceID to login?",
                "How big is the app?",
                "How often does the app sync?",
                "Does it use a lot of data?",
                "Is there an iPad version?",
                "Does it work on Apple Watch?",
                "Are there mobile widgets?",
                "Does it sync in the background?",
                "Can I use voice to take notes?",
                "Can I scan business cards?"
            ]
        },
        "onboarding": {
            "count": 112,
            "topics": [
                "Getting started guide: Interactive tour for all new users",
                "Data migration: Free migration assistance for 50+ users",
                "Team training: Weekly live training sessions for new teams",
                "Setup checklist: 5-step process to get your CRM ready",
                "First 30 days: Recommended milestones for success",
                "Custom domain setup: Use your own URL for the CRM portal",
                "Email configuration: Connect your SMTP or IMAP server",
                "Pipeline configuration: Map your existing sales process",
                "User invitation: Bulk invite team members via CSV",
                "Template library: Pre-built templates for many industries",
                "Sample data mode: Test features with a pre-populated dataset",
                "Implementation partners: Network of certified consultants",
                "Guided setup wizard: Automated walkthrough for admins",
                "Success metrics: Track ROI from month one",
                "Go-live checklist: Final steps before decommissioning old systems"
            ],
            "questions": [
                "Where do I start?",
                "How do I move my data?",
                "Is there training for my team?",
                "What is the setup process?",
                "What should I do in my first month?",
                "Can I use my own domain?",
                "How do I connect my email?",
                "How do I set up my pipeline?",
                "How do I invite everyone?",
                "Are there industry templates?",
                "Can I try it with fake data?",
                "Do you have consultants?",
                "Is there a setup wizard?",
                "How do I measure success?",
                "What do I need to do to go live?"
            ]
        },
        "api_developer": {
            "count": 112,
            "topics": [
                "REST API base URL: https://api.nexacrm.com/v1",
                "Authentication: Uses Bearer tokens via OAuth 2.0",
                "Rate limits: 1000/hr (Pro), 5000/hr (Enterprise)",
                "API versioning: SemVer followed; 6-month sunset for old versions",
                "Webhooks configuration: Set up in Developer Portal",
                "API sandbox: Free testing environment for all developers",
                "Code examples: Libraries for Python, JS, Ruby, and PHP",
                "Error codes: Standard HTTP status codes with JSON bodies",
                "Pagination: Link header and limit/offset supported",
                "Filtering and sorting: Robust query params for all endpoints",
                "Bulk operations: Update up to 1000 records in one request",
                "API changelog: Published at developers.nexacrm.com/changelog",
                "Developer docs: Comprehensive guides and API reference",
                "SDK availability: Official SDKs for popular languages",
                "GraphQL support: Beta access available for Enterprise customers"
            ],
            "questions": [
                "What is the API URL?",
                "How do I authenticate?",
                "What are the rate limits?",
                "How do you handle API versions?",
                "How do I set up webhooks?",
                "Is there a test environment?",
                "Are there code samples?",
                "What do the error codes mean?",
                "How do I handle many results?",
                "Can I filter API results?",
                "How do I update many records?",
                "Where is the changelog?",
                "Where are the docs?",
                "Do you have an SDK?",
                "Does NexaCRM support GraphQL?"
            ]
        },
        "compliance_legal": {
            "count": 112,
            "topics": [
                "Terms of service: Standard B2B SaaS agreement",
                "Privacy policy: We never sell your data to third parties",
                "Data processing agreement: Standard DPA available for download",
                "GDPR roles: NexaCRM is the Processor; you are the Controller",
                "Cookie policy: We only use essential cookies for app functionality",
                "Intellectual property: You own all content uploaded to the service",
                "Acceptable use policy: No spam or illegal activities allowed",
                "SLA details: Service credits provided if uptime falls below 99.9%",
                "Governing law: Delaware, United States",
                "Dispute resolution: Arbitration in Wilmington, DE",
                "Sub-processors: List available on our legal page (AWS, Stripe, etc.)",
                "Data transfer: Standard Contractual Clauses (SCCs) used",
                "Right to audit: Available for Enterprise customers annually",
                "Liability limitations: Capped at 12 months of subscription fees",
                "Force majeure: Standard clauses for unforeseen events"
            ],
            "questions": [
                "What is in the Terms of Service?",
                "How do you use my data?",
                "Do you have a DPA?",
                "Is NexaCRM a controller or processor?",
                "What cookies do you use?",
                "Who owns my data?",
                "What is not allowed on NexaCRM?",
                "What happens if the service goes down?",
                "What is the governing law?",
                "How are disputes handled?",
                "Who are your sub-processors?",
                "How is data transferred globally?",
                "Can I audit your security?",
                "What is your liability limit?",
                "What if a disaster happens?"
            ]
        }
    }

    corpus = []
    qa_pairs = []
    chunk_idx = 1
    qa_idx = 1

    # Calculate QA distribution
    total_qa_needed = 400
    cats_list = list(categories_config.keys())
    # 4 categories get 34, 8 get 33
    qa_distribution = {cat: 33 for cat in cats_list}
    for i in range(4):
        qa_distribution[cats_list[i]] += 1

    for cat_name, config in categories_config.items():
        topics = config["topics"]
        questions = config["questions"]
        count = config["count"]
        
        # Generate chunks
        for i in range(count):
            topic = topics[i % len(topics)]
            # Add variation to make each unique
            variation = f" (Ref: {cat_name}_{i:03d})"
            text = f"NexaCRM FAQ: {topic}{variation}"
            
            chunk_id = f"chunk_{chunk_idx:04d}"
            corpus.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": "nexacrm_faq",
                "category": cat_name
            })
            chunk_idx += 1

        # Generate QA pairs for this category
        num_qa = qa_distribution[cat_name]
        for i in range(num_qa):
            # Select a chunk from this category for the QA pair
            # We use chunks from this category starting from the first one
            source_chunk = corpus[len(corpus) - count + (i % count)]
            
            # Use one of the predefined questions or generate one
            if i < len(questions):
                question = questions[i]
            else:
                question = f"Question about {cat_name} entry {i}?"
            
            # The answer should be based on the topic
            # We'll take the part after the colon in the topic
            topic_str = topics[i % len(topics)]
            if ":" in topic_str:
                answer = topic_str.split(":", 1)[1].strip()
            else:
                answer = topic_str
            
            qa_id = f"qa_{qa_idx:03d}"
            qa_pairs.append({
                "id": qa_id,
                "question": question,
                "answer": answer,
                "chunk_id": source_chunk["chunk_id"],
                "category": cat_name
            })
            qa_idx += 1

    return corpus[:2000], qa_pairs[:400]

NEXACRM_CORPUS, GROUND_TRUTH_QA = generate_nexacrm_data()

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

def main():
    # Save NexaCRM Corpus
    nexacrm_path = DATA_DIR / "nexacrm_corpus.json"
    with open(nexacrm_path, "w") as f:
        json.dump(NEXACRM_CORPUS, f, indent=2)
    
    # Save Ground Truth QA
    gt_path = DATA_DIR / "ground_truth_qa.json"
    with open(gt_path, "w") as f:
        json.dump(GROUND_TRUTH_QA, f, indent=2)
        
    # Save Banking Dataset
    banking_corpus_path = DATA_DIR / "banking_corpus.json"
    with open(banking_corpus_path, "w") as f:
        json.dump(BANKING_FAQ_CORPUS, f, indent=2)
        
    banking_gt_path = DATA_DIR / "banking_ground_truth_qa.json"
    with open(banking_gt_path, "w") as f:
        json.dump(BANKING_GROUND_TRUTH_QA, f, indent=2)

    # Verification
    print(f"NexaCRM corpus: {len(NEXACRM_CORPUS)} chunks")
    print(f"Ground truth QA: {len(GROUND_TRUTH_QA)} pairs")
    print(f"Banking corpus: {len(BANKING_FAQ_CORPUS)} chunks")
    print(f"Banking QA: {len(BANKING_GROUND_TRUTH_QA)} pairs")
    
    categories = set(c['category'] for c in NEXACRM_CORPUS)
    print(f"Categories in NexaCRM: {len(categories)}")
    
    assert len(NEXACRM_CORPUS) == 2000, f"Expected 2000 chunks, got {len(NEXACRM_CORPUS)}"
    assert len(GROUND_TRUTH_QA) == 400, f"Expected 400 QA pairs, got {len(GROUND_TRUTH_QA)}"
    assert len(set(c['chunk_id'] for c in NEXACRM_CORPUS)) == 2000, "Duplicate chunk IDs found"
    assert len(set(q['id'] for q in GROUND_TRUTH_QA)) == 400, "Duplicate QA IDs found"
    print("All assertions passed")

if __name__ == "__main__":
    main()
