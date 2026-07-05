SF_DOCS = [
    # Core Objects
    "Salesforce Object: Account. Fields: Id, Name, Industry, AnnualRevenue, BillingCity, BillingCountry, OwnerId, Phone, Website, Type, NumberOfEmployees, Description. Relationships: has many Contacts, Opportunities, Cases, Contracts.",
    "Salesforce Object: Contact. Fields: Id, FirstName, LastName, Email, Phone, AccountId, Title, Department, MailingCity, OwnerId, LeadSource. Relationships: belongs to Account, linked to Opportunities via OpportunityContactRole.",
    "Salesforce Object: Opportunity. Fields: Id, Name, AccountId, Amount, StageName, CloseDate, Probability, OwnerId, LeadSource, Type, ForecastCategory, CampaignId. Relationships: has many OpportunityLineItems, QuoteLineItems via Quote.",
    "Salesforce Object: Lead. Fields: Id, FirstName, LastName, Email, Company, Status, LeadSource, Industry, AnnualRevenue, Phone, OwnerId, IsConverted, ConvertedAccountId, ConvertedContactId, ConvertedOpportunityId.",
    "Salesforce Object: Case. Fields: Id, CaseNumber, Subject, Description, Status, Priority, AccountId, ContactId, OwnerId, Origin, Type, Reason, IsClosedOnCreate. Relationships: belongs to Account and Contact.",
    "Salesforce Object: Quote. Fields: Id, Name, OpportunityId, Status, ExpirationDate, TotalPrice, GrandTotal, Discount, ShippingHandling, Tax. Relationships: belongs to Opportunity, has many QuoteLineItems.",
    "Salesforce Object: QuoteLineItem. Fields: Id, QuoteId, Product2Id, Quantity, UnitPrice, TotalPrice, Discount, Description, PricebookEntryId. Relationships: belongs to Quote. To get QuoteLineItems for an Order: join via Quote.OpportunityId to Order.OpportunityId.",
    "Salesforce Object: Order. Fields: Id, AccountId, OpportunityId, ContractId, Status, EffectiveDate, EndDate, TotalAmount, OwnerId, BillingCity. Status values: Draft, Activated. Relationships: belongs to Account, linked to Opportunity.",
    "Salesforce Object: OrderItem. Fields: Id, OrderId, Product2Id, Quantity, UnitPrice, TotalPrice, PricebookEntryId. Relationships: belongs to Order.",
    "Salesforce Object: Product2. Fields: Id, Name, ProductCode, Description, IsActive, Family, StockKeepingUnit.",
    "Salesforce Object: PricebookEntry. Fields: Id, Pricebook2Id, Product2Id, UnitPrice, IsActive, UseStandardPrice. Relationships: links Product2 to a Pricebook.",
    "Salesforce Object: Contract. Fields: Id, AccountId, Status, StartDate, EndDate, ContractTerm, OwnerId. Status values: Draft, Activated, Expired.",
    "Salesforce Object: Task. Fields: Id, Subject, Status, Priority, WhoId, WhatId, OwnerId, ActivityDate, Description. WhoId links to Contact or Lead. WhatId links to any object.",
    "Salesforce Object: Event. Fields: Id, Subject, StartDateTime, EndDateTime, WhoId, WhatId, OwnerId, Location, Description.",
    "Salesforce Object: Campaign. Fields: Id, Name, Status, Type, StartDate, EndDate, BudgetedCost, ActualCost, ExpectedRevenue, NumberOfLeads, NumberOfConvertedLeads.",
    "Salesforce Object: CampaignMember. Fields: Id, CampaignId, LeadId, ContactId, Status, HasResponded.",

    # SOQL Syntax
    "SOQL SELECT syntax: SELECT field1, field2 FROM ObjectName WHERE condition ORDER BY field LIMIT n. Example: SELECT Id, Name, Amount FROM Opportunity WHERE StageName = 'Closed Won' ORDER BY Amount DESC LIMIT 10.",
    "SOQL relationship query (parent to child): SELECT Id, Name, (SELECT Id, LastName FROM Contacts) FROM Account. This is a subquery — gets Account with all related Contacts.",
    "SOQL relationship query (child to parent): SELECT Id, Name, Account.Name, Account.Industry FROM Contact. Use dot notation to traverse parent relationships.",
    "SOQL WHERE operators: =, !=, <, >, <=, >=, LIKE, IN, NOT IN, INCLUDES, EXCLUDES. Example: WHERE StageName IN ('Closed Won', 'Proposal') AND Amount > 50000.",
    "SOQL date literals: TODAY, YESTERDAY, THIS_WEEK, LAST_WEEK, THIS_MONTH, LAST_MONTH, THIS_QUARTER, LAST_QUARTER, THIS_YEAR, LAST_YEAR, LAST_N_DAYS:n. Example: WHERE CloseDate = THIS_QUARTER.",
    "SOQL aggregate functions: COUNT(), SUM(), AVG(), MIN(), MAX(). Example: SELECT StageName, SUM(Amount), COUNT(Id) FROM Opportunity GROUP BY StageName.",
    "SOQL LIKE operator: % is wildcard. Example: WHERE Name LIKE '%Salesforce%' finds records containing 'Salesforce'. WHERE Name LIKE 'S%' finds records starting with S.",
    "SOQL NULL checks: WHERE Field = null finds records where field is empty. WHERE Field != null finds records where field has a value.",
    "SOQL ORDER BY and LIMIT: SELECT Id, Name FROM Account ORDER BY Name ASC LIMIT 50 OFFSET 100. OFFSET skips records for pagination.",
    "SOQL FOR UPDATE: locks records during query. SELECT Id FROM Account WHERE Id = '001...' FOR UPDATE. Used in Apex triggers to prevent race conditions.",

    # Common SOQL Recipes
    "SOQL: Get QuoteLineItems for an Order. SELECT qli.Id, qli.Product2.Name, qli.Quantity, qli.UnitPrice FROM QuoteLineItem qli WHERE qli.Quote.OpportunityId IN (SELECT OpportunityId FROM Order WHERE Id = 'ORDER_ID').",
    "SOQL: Get all open Opportunities closing this quarter. SELECT Id, Name, Amount, StageName, CloseDate, Account.Name FROM Opportunity WHERE IsClosed = false AND CloseDate = THIS_QUARTER ORDER BY Amount DESC.",
    "SOQL: Get Contacts for an Account. SELECT Id, FirstName, LastName, Email, Title FROM Contact WHERE AccountId = 'ACCOUNT_ID' ORDER BY LastName.",
    "SOQL: Count deals by stage. SELECT StageName, COUNT(Id) dealCount, SUM(Amount) totalAmount FROM Opportunity WHERE IsClosed = false GROUP BY StageName.",
    "SOQL: Get Cases by priority with Account name. SELECT Id, CaseNumber, Subject, Priority, Status, Account.Name, Contact.Email FROM Case WHERE Status != 'Closed' ORDER BY Priority.",
    "SOQL: Find duplicate leads by email. SELECT Email, COUNT(Id) FROM Lead WHERE IsConverted = false GROUP BY Email HAVING COUNT(Id) > 1.",
    "SOQL: Get Campaign performance. SELECT Id, Name, NumberOfLeads, NumberOfConvertedLeads, ActualCost, ExpectedRevenue FROM Campaign WHERE Status = 'Active'.",
    "SOQL: Opportunities without activities in 30 days. SELECT Id, Name, Amount, LastActivityDate FROM Opportunity WHERE IsClosed = false AND LastActivityDate < LAST_N_DAYS:30.",
    "SOQL: Get Order with OrderItems and Product names. SELECT Id, OrderNumber, TotalAmount, (SELECT Id, Product2.Name, Quantity, UnitPrice FROM OrderItems) FROM Order WHERE Status = 'Activated'.",
    "SOQL: Accounts with no Opportunities. SELECT Id, Name FROM Account WHERE Id NOT IN (SELECT AccountId FROM Opportunity WHERE AccountId != null).",

    # Salesforce Relationships
    "Salesforce relationship: Account → Opportunity is one-to-many. One Account can have many Opportunities. AccountId on Opportunity is the lookup field.",
    "Salesforce relationship: Opportunity → Quote is one-to-many. One Opportunity can have many Quotes. OpportunityId on Quote links them.",
    "Salesforce relationship: Quote → QuoteLineItem is one-to-many. QuoteId on QuoteLineItem links them. Each QuoteLineItem is one product on the quote.",
    "Salesforce relationship: Order → OrderItem is one-to-many. OrderId on OrderItem links them. To get products on an order use OrderItems subquery.",
    "Salesforce relationship: Order → Opportunity. OrderId has OpportunityId field. To get QuoteLineItems for an Order, go Order.OpportunityId → Quote.OpportunityId → QuoteLineItem.QuoteId.",
    "Salesforce relationship: Contact → Account is many-to-one. A Contact belongs to one Account via AccountId. An Account can have many Contacts.",
    "Salesforce relationship: Case → Account and Contact. Case has both AccountId and ContactId lookups. A Case can be linked to both.",

    # Salesforce Concepts
    "Salesforce Governor Limits: SOQL queries limited to 100 per transaction (synchronous), 200 (asynchronous). Total records returned: 50,000. Use LIMIT clause and selective WHERE filters to stay within limits.",
    "Salesforce Bulk API: for processing more than 50,000 records. Uses jobs and batches. Best for data migration, mass updates, and large exports. Supports CSV, JSON, XML.",
    "Salesforce Apex Trigger: executes before or after DML operations (insert, update, delete). Best practice: one trigger per object, delegate to handler class. Use trigger.new and trigger.old collections.",
    "Salesforce Flow: declarative automation tool. Types: Screen Flow (user-facing), Record-Triggered Flow (replaces workflow rules/process builder), Schedule-Triggered Flow, Platform Event Flow.",
    "Salesforce Lightning Web Component (LWC): modern JavaScript framework for building Salesforce UI. Uses standard web components. Can be embedded in App Pages, Record Pages, Experience Cloud.",
    "Salesforce Einstein Analytics / CRM Analytics: Salesforce's BI tool. Uses SAQL (Salesforce Analytics Query Language) instead of SOQL. Connects to Salesforce objects and external data.",
    "Salesforce Sandbox types: Developer (200MB), Developer Pro (1GB), Partial Copy (5GB, sample of prod data), Full (complete copy of production). Used for testing before deploying to production.",
    "Salesforce deployment: use Change Sets (declarative), Salesforce CLI with source tracking (dx), or tools like Copado/Gearset for CI/CD pipelines.",
]


def get_sf_docs() -> list[str]:
    return SF_DOCS
