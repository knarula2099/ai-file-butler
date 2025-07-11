# demo/create_test_data.py
import os
import random
import string
from pathlib import Path
from datetime import datetime, timedelta
import json

class TestDataGenerator:
    """Generates realistic test data for demonstrating the AI File Butler"""
    
    def __init__(self, base_path: str = "demo/test_scenarios"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_messy_downloads_scenario(self) -> str:
        """Create a realistic messy downloads folder"""
        scenario_path = self.base_path / "messy_downloads"
        scenario_path.mkdir(exist_ok=True)
        
        print(f"Creating messy downloads scenario in: {scenario_path}")
        
        # Generate various types of files
        self._create_documents(scenario_path)
        self._create_images(scenario_path)
        self._create_software_files(scenario_path)
        self._create_media_files(scenario_path)
        self._create_random_junk(scenario_path)
        self._create_duplicates(scenario_path)
        
        # Create some nested mess
        nested_folder = scenario_path / "New folder (2)"
        nested_folder.mkdir(exist_ok=True)
        self._create_more_random_files(nested_folder)
        
        print(f"Created {len(list(scenario_path.rglob('*')))} files and folders")
        return str(scenario_path)
    
    def create_photo_collection_scenario(self) -> str:
        """Create a photo collection that needs organization"""
        scenario_path = self.base_path / "photo_collection"
        scenario_path.mkdir(exist_ok=True)
        
        print(f"Creating photo collection scenario in: {scenario_path}")
        
        # Create photos with different naming patterns
        self._create_camera_photos(scenario_path)
        self._create_screenshot_photos(scenario_path)
        self._create_social_media_photos(scenario_path)
        self._create_vacation_photos(scenario_path)
        
        print(f"Created {len(list(scenario_path.rglob('*')))} photo files")
        return str(scenario_path)
    
    def create_document_archive_scenario(self) -> str:
        """Create a document collection for organization"""
        scenario_path = self.base_path / "document_archive"
        scenario_path.mkdir(exist_ok=True)
        
        print(f"Creating document archive scenario in: {scenario_path}")
        
        self._create_work_documents(scenario_path)
        self._create_personal_documents(scenario_path)
        self._create_financial_documents(scenario_path)
        self._create_educational_documents(scenario_path)
        
        print(f"Created {len(list(scenario_path.rglob('*')))} document files")
        return str(scenario_path)
    
    def _create_documents(self, path: Path):
        """Create various document files"""
        documents = [
            ("Meeting_Notes_Q3_2024.txt", self._generate_meeting_notes()),
            ("Project_Proposal_Draft.txt", self._generate_project_proposal()),
            ("Budget_Analysis_Final.txt", self._generate_budget_analysis()),
            ("resume_john_doe_2024.pdf", ""),  # Empty for demo
            ("invoice_12345.pdf", ""),
            ("contract_template.docx", ""),
            ("README.md", self._generate_readme()),
            ("todo_list.txt", self._generate_todo_list()),
            ("random_notes.txt", self._generate_random_notes()),
        ]
        
        for filename, content in documents:
            file_path = path / filename
            if content and filename.endswith('.txt'):
                file_path.write_text(content)
            else:
                file_path.touch()
            
            # Randomize modification times
            self._randomize_file_time(file_path)
    
    def _create_images(self, path: Path):
        """Create image files with various naming patterns"""
        images = [
            "IMG_20240315_143022.jpg",
            "Screenshot_2024-03-10_at_2.15.34_PM.png",
            "vacation_beach_sunset.jpeg",
            "profile_picture_new.png",
            "meme_funny_cat.gif",
            "diagram_architecture.svg",
            "invoice_scan.jpg",
            "receipt_grocery_store.png",
            "family_photo_christmas_2023.jpg",
            "DCIM_0001.jpg",
            "DCIM_0002.jpg",
            "DCIM_0003.jpg",  # Potential duplicates
        ]
        
        for filename in images:
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_software_files(self, path: Path):
        """Create software-related files"""
        software_files = [
            "setup_chrome_installer.exe",
            "python-3.9.7-installer.msi",
            "adobe_photoshop_trial.dmg",
            "driver_update_v2.1.zip",
            "game_save_backup.zip",
            "project_source_code.tar.gz",
            "database_backup_20240301.sql",
            "app_config.json",
        ]
        
        for filename in software_files:
            file_path = path / filename
            if filename.endswith('.json'):
                content = {"version": "1.0", "settings": {"theme": "dark"}}
                file_path.write_text(json.dumps(content, indent=2))
            else:
                file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_media_files(self, path: Path):
        """Create media files"""
        media_files = [
            "song_favorite_track.mp3",
            "podcast_episode_123.mp3",
            "video_tutorial_python.mp4",
            "movie_trailer_2024.avi",
            "presentation_recording.mov",
            "audio_note_meeting.wav",
        ]
        
        for filename in media_files:
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_random_junk(self, path: Path):
        """Create random junk files that often accumulate"""
        junk_files = [
            "Untitled.txt",
            "New Text Document.txt",
            "Copy of Copy of document.txt",
            "temp_file_delete_later.tmp",
            "~$document.docx",  # Word temp file
            ".DS_Store",  # Mac system file
            "Thumbs.db",  # Windows thumbnail cache
            "desktop.ini",  # Windows system file
            "random_string_" + ''.join(random.choices(string.ascii_lowercase, k=8)) + ".txt",
        ]
        
        for filename in junk_files:
            file_path = path / filename
            if filename.endswith('.txt'):
                file_path.write_text("This is a temporary file that can probably be deleted.")
            else:
                file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_duplicates(self, path: Path):
        """Create duplicate files for testing duplicate detection"""
        original_content = "This is the original content for duplicate testing."
        
        # Create original and duplicates
        original = path / "important_document.txt"
        original.write_text(original_content)
        
        # Exact duplicates
        duplicate1 = path / "important_document_copy.txt"
        duplicate1.write_text(original_content)
        
        duplicate2 = path / "important_document (1).txt"
        duplicate2.write_text(original_content)
        
        # Near duplicate (slightly different content)
        near_duplicate = path / "important_document_edited.txt"
        near_duplicate.write_text(original_content + "\n\nThis line was added later.")
        
        for file_path in [original, duplicate1, duplicate2, near_duplicate]:
            self._randomize_file_time(file_path)
    
    def _create_more_random_files(self, path: Path):
        """Create additional random files in subdirectories"""
        files = [
            "backup_file.bak",
            "log_file.log",
            "data_export.csv",
            "configuration.ini",
            "script.py",
        ]
        
        for filename in files:
            file_path = path / filename
            if filename.endswith('.py'):
                file_path.write_text("# This is a Python script\nprint('Hello, World!')")
            elif filename.endswith('.csv'):
                file_path.write_text("name,age,city\nJohn,30,New York\nJane,25,Los Angeles")
            else:
                file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_camera_photos(self, path: Path):
        """Create camera-style photos"""
        for i in range(1, 16):
            filename = f"IMG_{20240300 + i}_{random.randint(100000, 999999)}.jpg"
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_screenshot_photos(self, path: Path):
        """Create screenshot-style photos"""
        screenshots = [
            "Screenshot 2024-03-15 at 10.30.45.png",
            "Screen Shot 2024-03-16 at 2.15.30 PM.png",
            "Screenshot_20240317_143022.png",
            "screenshot_error_message.png",
            "screenshot_funny_tweet.png",
        ]
        
        for filename in screenshots:
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_social_media_photos(self, path: Path):
        """Create social media style photos"""
        social_photos = [
            "instagram_story_beach.jpg",
            "facebook_profile_update.jpg",
            "linkedin_headshot_professional.jpg",
            "twitter_meme_save.png",
            "snapchat_filter_funny.jpg",
        ]
        
        for filename in social_photos:
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_vacation_photos(self, path: Path):
        """Create vacation-themed photos"""
        vacation_photos = [
            "hawaii_trip_2024_sunset.jpg",
            "paris_eiffel_tower.jpg",
            "tokyo_sushi_dinner.jpg",
            "london_big_ben.jpg",
            "vacation_group_photo.jpg",
        ]
        
        for filename in vacation_photos:
            file_path = path / filename
            file_path.touch()
            self._randomize_file_time(file_path)
    
    def _create_work_documents(self, path: Path):
        """Create work-related documents"""
        work_docs = [
            ("quarterly_report_q1_2024.txt", self._generate_quarterly_report()),
            ("team_meeting_agenda.txt", self._generate_meeting_agenda()),
            ("project_timeline.txt", self._generate_project_timeline()),
            ("employee_handbook.txt", self._generate_handbook()),
            ("performance_review_template.txt", self._generate_performance_review()),
        ]
        
        for filename, content in work_docs:
            file_path = path / filename
            file_path.write_text(content)
            self._randomize_file_time(file_path)
    
    def _create_personal_documents(self, path: Path):
        """Create personal documents"""
        personal_docs = [
            ("recipe_chocolate_cake.txt", self._generate_recipe()),
            ("workout_routine.txt", self._generate_workout()),
            ("book_recommendations.txt", self._generate_book_list()),
            ("vacation_planning_notes.txt", self._generate_vacation_plan()),
            ("grocery_list.txt", self._generate_grocery_list()),
        ]
        
        for filename, content in personal_docs:
            file_path = path / filename
            file_path.write_text(content)
            self._randomize_file_time(file_path)
    
    def _create_financial_documents(self, path: Path):
        """Create financial documents"""
        financial_docs = [
            ("budget_2024.txt", self._generate_budget()),
            ("tax_deductions_list.txt", self._generate_tax_info()),
            ("investment_portfolio_notes.txt", self._generate_investment_notes()),
            ("monthly_expenses_march.txt", self._generate_expenses()),
        ]
        
        for filename, content in financial_docs:
            file_path = path / filename
            file_path.write_text(content)
            self._randomize_file_time(file_path)
    
    def _create_educational_documents(self, path: Path):
        """Create educational documents"""
        edu_docs = [
            ("python_programming_notes.txt", self._generate_programming_notes()),
            ("machine_learning_concepts.txt", self._generate_ml_notes()),
            ("data_science_resources.txt", self._generate_resources()),
            ("online_course_certificate.txt", self._generate_certificate()),
        ]
        
        for filename, content in edu_docs:
            file_path = path / filename
            file_path.write_text(content)
            self._randomize_file_time(file_path)
    
    def _randomize_file_time(self, file_path: Path):
        """Randomize file modification time within the last year"""
        now = datetime.now()
        random_days = random.randint(1, 365)
        random_time = now - timedelta(days=random_days)
        timestamp = random_time.timestamp()
        
        os.utime(file_path, (timestamp, timestamp))
    
    # Content generation methods
    def _generate_meeting_notes(self) -> str:
        return """Meeting Notes - Q3 Planning Session
Date: March 15, 2024
Attendees: John, Sarah, Mike, Lisa

Agenda:
1. Review Q2 performance
2. Set Q3 objectives
3. Resource allocation
4. Timeline discussion

Key Points:
- Q2 exceeded targets by 15%
- Focus on new product launch in Q3
- Need additional developer resources
- Marketing campaign to start in July

Action Items:
- John: Prepare budget proposal by March 20
- Sarah: Draft marketing plan by March 25
- Mike: Technical requirements document by March 22

Next Meeting: March 29, 2024"""
    
    def _generate_project_proposal(self) -> str:
        return """Project Proposal: Customer Data Analytics Platform

Executive Summary:
This proposal outlines the development of a comprehensive customer data analytics platform to improve our understanding of customer behavior and drive business growth.

Objectives:
- Centralize customer data from multiple sources
- Provide real-time analytics and insights
- Enable data-driven decision making
- Improve customer experience through personalization

Technical Requirements:
- Cloud-based infrastructure (AWS/Azure)
- Data integration capabilities
- Machine learning algorithms for predictive analytics
- User-friendly dashboard interface

Timeline: 6 months
Budget: $150,000
Team: 4 developers, 1 data scientist, 1 project manager

Expected ROI: 25% increase in customer retention within first year"""
    
    def _generate_budget_analysis(self) -> str:
        return """Budget Analysis Report - FY 2024

Revenue Projections:
Q1: $1.2M (actual: $1.35M - 12% above target)
Q2: $1.4M (forecast)
Q3: $1.6M (forecast)
Q4: $1.8M (forecast)

Expense Categories:
Personnel: 60% of budget
Technology: 20% of budget
Marketing: 15% of budget
Operations: 5% of budget

Key Findings:
- Personnel costs under control
- Technology investments showing ROI
- Marketing efficiency improved by 8%
- Need to optimize operational expenses

Recommendations:
1. Increase technology budget by 5%
2. Implement cost-saving measures in operations
3. Continue current marketing strategy
4. Plan for strategic hiring in Q4"""
    
    def _generate_readme(self) -> str:
        return """# Project README

## Description
This is a sample project demonstrating various programming concepts and best practices.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from main import App
app = App()
app.run()
```

## Features
- Feature 1: Data processing
- Feature 2: API integration
- Feature 3: User interface

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details."""
    
    def _generate_todo_list(self) -> str:
        return """TODO List - March 2024

Work:
- [ ] Finish quarterly report
- [x] Update project documentation
- [ ] Schedule team meeting
- [ ] Review code submissions
- [x] Attend client presentation

Personal:
- [ ] Book vacation flights
- [x] Renew car insurance
- [ ] Call dentist for appointment
- [ ] Organize home office
- [ ] Start reading new book

Shopping:
- [x] Groceries for the week
- [ ] New laptop charger
- [ ] Birthday gift for mom
- [ ] Running shoes

Weekend:
- [ ] Clean garage
- [ ] Plant spring garden
- [ ] Visit museum exhibition
- [x] Family dinner on Sunday"""
    
    def _generate_random_notes(self) -> str:
        return """Random Notes and Ideas

Meeting with client tomorrow:
- Bring contract documents
- Discuss timeline changes
- Address budget concerns

Ideas for blog post:
- "10 Tips for Better Time Management"
- Focus on practical strategies
- Include personal examples
- Add actionable takeaways

Book recommendations from Sarah:
- "Atomic Habits" by James Clear
- "The Power of Now" by Eckhart Tolle
- "Sapiens" by Yuval Noah Harari

Recipe modifications for dinner party:
- Double the appetizer quantities
- Add vegetarian option for main course
- Prepare dessert day before
- Create music playlist

Random thought: Maybe learn Spanish this year?
Check out language learning apps, find local classes."""
    
    def _generate_quarterly_report(self) -> str:
        return """Quarterly Business Report - Q1 2024

Executive Summary:
Q1 2024 has been a strong quarter with revenue growth of 18% compared to Q1 2023. Key achievements include successful product launch, expansion into new markets, and improved customer satisfaction scores.

Financial Performance:
- Revenue: $2.4M (vs. $2.1M target)
- Gross Margin: 68%
- Net Profit: $480K
- Operating Expenses: $1.15M

Key Metrics:
- Customer Acquisition: 1,200 new customers
- Customer Retention: 94%
- Average Order Value: $350
- Customer Satisfaction: 4.7/5.0

Market Analysis:
The market continues to show strong demand for our products. Competitors have introduced similar offerings, but our unique value proposition maintains our competitive advantage.

Challenges:
- Supply chain disruptions in March
- Increased marketing costs
- Talent acquisition in technical roles

Opportunities:
- Partnership discussions with three major clients
- Potential expansion into European markets
- New product development pipeline strong

Q2 Outlook:
Expected revenue growth of 15-20% with focus on operational efficiency and customer experience improvements."""
    
    def _generate_meeting_agenda(self) -> str:
        return """Team Meeting Agenda - March 28, 2024
Time: 10:00 AM - 11:30 AM
Location: Conference Room B / Zoom

1. Welcome and Introductions (5 min)

2. Review of Action Items from Last Meeting (10 min)
   - Website redesign progress (Sarah)
   - Database optimization results (Mike)
   - Customer feedback analysis (Lisa)

3. Current Project Updates (30 min)
   - Project Alpha status and blockers
   - Project Beta timeline review
   - Resource allocation discussion

4. New Business (20 min)
   - Q2 planning kickoff
   - Team training opportunities
   - Process improvement suggestions

5. Team Announcements (10 min)
   - Upcoming holidays and PTO
   - New team member introduction
   - Office policy updates

6. Next Steps and Action Items (10 min)

7. Questions and Open Discussion (5 min)

Next Meeting: April 4, 2024
Prepared by: John Smith, Team Lead"""
    
    def _generate_project_timeline(self) -> str:
        return """Project Alpha Timeline - Customer Portal Development

Phase 1: Planning and Design (Weeks 1-3)
- Week 1: Requirements gathering and stakeholder interviews
- Week 2: Technical architecture design
- Week 3: UI/UX mockups and user flow design

Phase 2: Development (Weeks 4-10)
- Week 4-5: Backend API development
- Week 6-7: Database schema and integration
- Week 8-9: Frontend development
- Week 10: Integration and initial testing

Phase 3: Testing and Quality Assurance (Weeks 11-13)
- Week 11: Unit testing and code review
- Week 12: Integration testing and bug fixes
- Week 13: User acceptance testing

Phase 4: Deployment and Launch (Weeks 14-16)
- Week 14: Production environment setup
- Week 15: Soft launch with limited users
- Week 16: Full launch and monitoring

Key Milestones:
- March 30: Design approval
- April 20: MVP completion
- May 10: Testing completion
- May 25: Production launch

Risk Factors:
- Third-party API dependencies
- Resource availability during April
- Potential scope changes from stakeholders

Success Metrics:
- User adoption rate > 70% within first month
- Page load times < 2 seconds
- Customer satisfaction score > 4.5/5"""
    
    def _generate_handbook(self) -> str:
        return """Employee Handbook - Key Policies

Welcome to TechCorp! This handbook contains important information about company policies and procedures.

Work Hours and Flexibility:
- Standard hours: 9:00 AM - 5:00 PM
- Flexible start time: 8:00 AM - 10:00 AM
- Remote work: Up to 3 days per week
- Core collaboration hours: 10:00 AM - 3:00 PM

Time Off Policies:
- Vacation: 20 days per year (increasing with tenure)
- Sick leave: 10 days per year
- Personal days: 3 days per year
- Holidays: 12 company holidays plus floating holidays

Professional Development:
- Annual training budget: $2,000 per employee
- Conference attendance encouraged
- Internal lunch-and-learn sessions
- Mentorship program available

Code of Conduct:
- Treat all colleagues with respect and professionalism
- Maintain confidentiality of company and client information
- Report any concerns to HR or management
- Follow all safety protocols

Benefits Overview:
- Health insurance (company pays 80%)
- Dental and vision coverage
- 401(k) with 4% company match
- Life insurance
- Employee assistance program

Contact Information:
HR Department: hr@techcorp.com
IT Support: it@techcorp.com"""
    
    def _generate_performance_review(self) -> str:
        return """Performance Review Template

Employee Name: [To be filled]
Review Period: [Quarter/Year]
Reviewer: [Manager Name]
Date: [Review Date]

Performance Objectives:
1. Technical Skills Development
   - Goal: Complete advanced certification
   - Status: [In Progress/Completed/Not Started]
   - Comments: [Manager feedback]

2. Project Leadership
   - Goal: Lead at least one major project
   - Status: [In Progress/Completed/Not Started]
   - Comments: [Manager feedback]

3. Team Collaboration
   - Goal: Mentor junior team members
   - Status: [In Progress/Completed/Not Started]
   - Comments: [Manager feedback]

Strengths:
- [List key strengths and accomplishments]
- [Specific examples of excellent work]
- [Areas where employee excels]

Areas for Improvement:
- [Constructive feedback on growth areas]
- [Specific skills to develop]
- [Suggested training or resources]

Overall Rating:
[ ] Exceeds Expectations
[ ] Meets Expectations
[ ] Needs Improvement
[ ] Unsatisfactory

Goals for Next Period:
1. [Specific goal with timeline]
2. [Specific goal with timeline]
3. [Specific goal with timeline]

Employee Comments:
[Space for employee self-assessment and feedback]

Manager Signature: _______________ Date: _______
Employee Signature: _____________ Date: _______"""
    
    def _generate_recipe(self) -> str:
        return """Chocolate Cake Recipe

Ingredients:
- 2 cups all-purpose flour
- 2 cups granulated sugar
- 3/4 cup unsweetened cocoa powder
- 2 teaspoons baking powder
- 1 1/2 teaspoons baking soda
- 1 teaspoon salt
- 2 eggs
- 1 cup buttermilk
- 1/2 cup vegetable oil
- 2 teaspoons vanilla extract
- 1 cup hot coffee

For Frosting:
- 1/2 cup butter, softened
- 2/3 cup cocoa powder
- 3 cups powdered sugar
- 1/3 cup milk
- 1 teaspoon vanilla extract

Instructions:
1. Preheat oven to 350°F (175°C). Grease two 9-inch round pans.
2. In large bowl, mix flour, sugar, cocoa, baking powder, baking soda, and salt.
3. Add eggs, buttermilk, oil, and vanilla. Beat on medium speed for 2 minutes.
4. Stir in hot coffee (batter will be thin).
5. Pour into prepared pans.
6. Bake 30-35 minutes or until toothpick comes out clean.
7. Cool completely before frosting.

For frosting: Beat butter until fluffy. Add cocoa, then alternately add powdered sugar and milk. Beat until smooth.

Prep time: 20 minutes
Bake time: 35 minutes
Serves: 12

Notes: Can substitute coffee with hot water if preferred. Cake freezes well unfrosted."""
    
    def _generate_workout(self) -> str:
        return """Weekly Workout Routine

Monday - Upper Body Strength:
- Warm-up: 10 minutes cardio
- Push-ups: 3 sets of 12-15 reps
- Pull-ups: 3 sets of 8-10 reps
- Dumbbell rows: 3 sets of 12 reps
- Shoulder press: 3 sets of 10 reps
- Bicep curls: 3 sets of 12 reps
- Tricep dips: 3 sets of 10 reps
- Cool-down: 5 minutes stretching

Tuesday - Cardio:
- 30-45 minutes running or cycling
- Target heart rate: 70-80% max
- Include 5-minute warm-up and cool-down

Wednesday - Lower Body Strength:
- Warm-up: 10 minutes light cardio
- Squats: 3 sets of 15 reps
- Lunges: 3 sets of 12 reps per leg
- Deadlifts: 3 sets of 10 reps
- Calf raises: 3 sets of 15 reps
- Leg press: 3 sets of 12 reps
- Cool-down: 10 minutes stretching

Thursday - Active Recovery:
- 30 minutes yoga or light walking
- Focus on flexibility and mobility

Friday - Full Body Circuit:
- 4 rounds of:
  - Burpees: 10 reps
  - Mountain climbers: 20 reps
  - Plank: 30 seconds
  - Jumping jacks: 15 reps
  - Rest: 1 minute between rounds

Weekend:
- Choose fun activities: hiking, swimming, sports
- Listen to your body and rest if needed

Notes:
- Stay hydrated throughout workouts
- Progress gradually to avoid injury
- Get adequate sleep for recovery"""
    
    def _generate_book_list(self) -> str:
        return """Book Recommendations 2024

Fiction:
1. "The Seven Husbands of Evelyn Hugo" by Taylor Jenkins Reid
   - Captivating story about a reclusive Hollywood icon
   - Rating: 5/5 stars

2. "Where the Crawdads Sing" by Delia Owens
   - Beautiful coming-of-age story set in North Carolina
   - Rating: 4.5/5 stars

3. "The Midnight Library" by Matt Haig
   - Philosophical novel about life's infinite possibilities
   - Rating: 4/5 stars

Non-Fiction:
1. "Atomic Habits" by James Clear
   - Practical guide to building good habits
   - Life-changing insights on behavior change
   - Rating: 5/5 stars

2. "Educated" by Tara Westover
   - Powerful memoir about education and family
   - Thought-provoking and inspiring
   - Rating: 5/5 stars

3. "Sapiens" by Yuval Noah Harari
   - Fascinating look at human history and evolution
   - Makes you think differently about humanity
   - Rating: 4.5/5 stars

Business/Self-Development:
1. "The 7 Habits of Highly Effective People" by Stephen Covey
   - Timeless principles for personal effectiveness
   - Rating: 4.5/5 stars

2. "Mindset" by Carol Dweck
   - Revolutionary insights on growth vs. fixed mindset
   - Rating: 4/5 stars

Currently Reading:
- "The Power of Now" by Eckhart Tolle
- "Becoming" by Michelle Obama

Want to Read:
- "Dune" by Frank Herbert
- "The Lean Startup" by Eric Ries
- "Thinking, Fast and Slow" by Daniel Kahneman"""
    
    def _generate_vacation_plan(self) -> str:
        return """Japan Vacation Planning - April 2024

Trip Duration: 10 days (April 15-25, 2024)
Budget: $4,000 per person

Itinerary:
Days 1-3: Tokyo
- Arrival at Narita Airport
- Hotel: Shibuya District
- Must-see: Senso-ji Temple, Tokyo Skytree, Meiji Shrine
- Food: Try authentic ramen, sushi at Tsukiji Market
- Shopping: Harajuku, Ginza

Days 4-5: Kyoto
- Travel by bullet train (JR Pass)
- Hotel: Traditional ryokan with onsen
- Must-see: Fushimi Inari Shrine, Bamboo Grove, Golden Pavilion
- Experience: Tea ceremony, kimono rental

Days 6-7: Osaka
- Hotel: Dotonbori area
- Must-see: Osaka Castle, Universal Studios Japan
- Food: Street food tour, takoyaki, okonomiyaki
- Day trip to Nara to see deer park

Days 8-10: Return to Tokyo
- Last-minute shopping
- Visit any missed attractions
- Departure preparations

Packing List:
- Comfortable walking shoes
- Portable phone charger
- Cash (Japan is cash-heavy)
- JR Pass (buy before departure)
- Universal adapter
- Light layers for changing weather

Important Notes:
- Download Google Translate app
- Learn basic Japanese phrases
- Book restaurants in advance
- Carry cash for small establishments
- Respect local customs and etiquette

Emergency Contacts:
- Embassy: +81-3-3224-5000
- Travel insurance: Policy #ABC123
- Hotel contact numbers: [to be filled]

Estimated Costs:
- Flights: $1,200
- Hotels: $1,500
- Food: $800
- Transportation: $300
- Activities: $200"""
    
    def _generate_grocery_list(self) -> str:
        return """Grocery List - Week of March 25, 2024

Proteins:
- Chicken breast (2 lbs)
- Salmon fillets (1 lb)
- Ground turkey (1 lb)
- Eggs (1 dozen)
- Greek yogurt (large container)
- Almonds (1 bag)

Vegetables:
- Spinach (2 bags)
- Broccoli (2 heads)
- Bell peppers (red, yellow, green)
- Carrots (2 lbs bag)
- Onions (3 lbs bag)
- Garlic (1 bulb)
- Avocados (4 count)
- Tomatoes (6 count)

Fruits:
- Bananas (6 count)
- Apples (Honeycrisp, 6 count)
- Blueberries (2 containers)
- Oranges (6 count)
- Lemons (3 count)

Pantry Staples:
- Brown rice (2 lb bag)
- Quinoa (1 lb)
- Whole wheat bread
- Oatmeal (large container)
- Olive oil
- Coconut oil
- Honey
- Black beans (2 cans)
- Diced tomatoes (3 cans)

Dairy:
- Milk (1/2 gallon, 2%)
- Cheese (cheddar block)
- Butter (1 lb)
- Plain Greek yogurt (small containers)

Household:
- Laundry detergent
- Dish soap
- Paper towels (2 rolls)
- Toilet paper (12 pack)

Special Items:
- Dark chocolate (70% cacao)
- Green tea bags
- Sparkling water (12 pack)

Estimated Total: $125

Store Notes:
- Check for sales on organic produce
- Use coupons for household items
- Buy store brand for basic staples
- Don't forget reusable bags!"""
    
    def _generate_budget(self) -> str:
        return """Personal Budget 2024

Monthly Income: $6,500 (after taxes)

Fixed Expenses:
- Rent/Mortgage: $1,800 (28%)
- Car payment: $450 (7%)
- Insurance (auto/health): $350 (5%)
- Phone: $80 (1%)
- Internet: $70 (1%)
- Utilities: $150 (2%)
Total Fixed: $2,900 (45%)

Variable Expenses:
- Groceries: $500 (8%)
- Gas: $200 (3%)
- Dining out: $300 (5%)
- Entertainment: $200 (3%)
- Clothing: $150 (2%)
- Personal care: $100 (2%)
- Miscellaneous: $200 (3%)
Total Variable: $1,650 (26%)

Savings & Investments:
- Emergency fund: $500 (8%)
- 401(k): $650 (10%) [with employer match]
- IRA: $500 (8%)
- Investments: $300 (5%)
Total Savings: $1,950 (31%)

Budget Summary:
- Total Expenses: $4,550 (70%)
- Total Savings: $1,950 (30%)
- Remaining: $0

Financial Goals 2024:
- Build emergency fund to $15,000
- Max out IRA contribution ($6,500)
- Increase 401(k) to 15% by year-end
- Pay extra $200/month toward car loan
- Save for vacation fund: $3,000

Notes:
- Review budget monthly
- Track expenses with app
- Look for ways to reduce variable costs
- Consider side income opportunities
- Automate savings transfers"""
    
    def _generate_tax_info(self) -> str:
        return """Tax Deductions Checklist - 2024

Work-Related Deductions:
- Home office expenses (if self-employed)
- Professional development courses: $1,200
- Business travel expenses
- Professional organization memberships: $300
- Work-related books and subscriptions: $150

Medical Expenses:
- Health insurance premiums (if self-employed)
- Out-of-pocket medical expenses over 7.5% of AGI
- Prescription medications
- Dental and vision care
- Medical travel expenses

Charitable Contributions:
- Cash donations to qualified charities: $2,400
- Clothing donations to Goodwill: $500
- Volunteer mileage: 14 cents per mile
- Keep all receipts and acknowledgment letters

Education:
- Student loan interest: up to $2,500
- Tuition and fees (American Opportunity Credit)
- Educational supplies for continuing education

Investment-Related:
- Investment advisor fees
- Safe deposit box rental
- Investment-related publications

State and Local Taxes:
- State income tax or sales tax (whichever is higher)
- Property taxes: $3,600
- Vehicle registration fees

Other Potential Deductions:
- Mortgage interest: $8,400
- PMI (private mortgage insurance)
- Points paid on mortgage
- Gambling losses (up to winnings)

Important Reminders:
- Keep receipts for everything
- Document business mileage
- Track charitable contributions throughout year
- Consider bunching deductions if near standard deduction limit
- Consult tax professional for complex situations

2024 Standard Deduction:
- Single: $14,600
- Married Filing Jointly: $29,200
- Head of Household: $21,900"""
    
    def _generate_investment_notes(self) -> str:
        return """Investment Portfolio Notes - 2024

Current Portfolio Allocation:
- Stocks: 70% ($35,000)
- Bonds: 20% ($10,000)
- Cash/Emergency Fund: 10% ($5,000)
Total Portfolio Value: $50,000

Individual Holdings:
Stock Investments (70%):
- S&P 500 Index Fund (VTI): 40% - $20,000
- International Developed Markets (VTIAX): 15% - $7,500
- Emerging Markets (VWO): 5% - $2,500
- Individual Stocks: 10% - $5,000
  * Apple (AAPL): $1,500
  * Microsoft (MSFT): $1,500
  * Amazon (AMZN): $1,000
  * Tesla (TSLA): $1,000

Bond Investments (20%):
- Total Bond Market Index (BND): 15% - $7,500
- Treasury I Bonds: 5% - $2,500

Performance Review:
YTD Return: +8.5% (vs S&P 500: +7.2%)
Best Performer: Apple (+15.3%)
Worst Performer: Tesla (-8.2%)

Investment Strategy:
- Dollar-cost averaging monthly: $1,000
- Rebalance quarterly
- Focus on low-cost index funds
- Maintain emergency fund separate from investments
- Target allocation adjustment as I age

2024 Goals:
- Increase portfolio to $60,000
- Add REIT exposure (5% allocation)
- Research ESG investment options
- Consider tax-loss harvesting opportunities
- Review and possibly increase 401(k) contribution

Risk Assessment:
- Time horizon: 30+ years until retirement
- Risk tolerance: Moderate to aggressive
- Current age: 28
- Income stability: Good

Notes:
- Continue learning about investing
- Read "A Random Walk Down Wall Street"
- Consider meeting with fee-only financial advisor
- Track expenses with Personal Capital
- Remember: time in market beats timing the market"""
    
    def _generate_expenses(self) -> str:
        return """Monthly Expenses - March 2024

Housing (35%):
- Rent: $1,800
- Utilities: $145
- Internet: $70
- Renters insurance: $25
Total Housing: $2,040

Transportation (15%):
- Car payment: $450
- Auto insurance: $125
- Gas: $180
- Maintenance: $75
- Parking: $50
Total Transportation: $880

Food (12%):
- Groceries: $420
- Dining out: $280
- Coffee/lunch at work: $120
Total Food: $820

Personal (8%):
- Health insurance: $200
- Phone: $80
- Gym membership: $45
- Personal care: $85
- Clothing: $110
Total Personal: $520

Entertainment (5%):
- Streaming services: $35
- Movies/events: $85
- Books: $25
- Hobbies: $95
Total Entertainment: $240

Miscellaneous (5%):
- Gifts: $75
- Charity donations: $150
- Emergency repairs: $120
- Pet expenses: $85
Total Miscellaneous: $430

Total Monthly Expenses: $4,930

Income vs. Expenses:
Monthly Income: $6,500
Monthly Expenses: $4,930
Available for Savings: $1,570

Savings Allocation:
- Emergency fund: $470
- 401(k): $650 (10% of gross)
- IRA: $450
Total Savings Rate: 24%

Notes:
- Dining out expenses higher than budgeted
- Need to find ways to reduce transportation costs
- Emergency repair was one-time expense
- Good month for staying within entertainment budget

Action Items:
- Cook more meals at home
- Research cheaper auto insurance
- Set up automatic transfers for savings
- Track daily expenses with app"""
    
    def _generate_programming_notes(self) -> str:
        return """Python Programming Study Notes

Data Structures:
1. Lists - Ordered, mutable collections
   - list.append(item) - adds to end
   - list.insert(index, item) - adds at position
   - list.remove(item) - removes first occurrence
   - List comprehensions: [x*2 for x in range(10)]

2. Dictionaries - Key-value pairs
   - dict.get(key, default) - safe access
   - dict.keys(), dict.values(), dict.items()
   - Dictionary comprehensions: {k: v for k, v in pairs}

3. Sets - Unordered, unique elements
   - set.add(item), set.remove(item)
   - Set operations: union, intersection, difference

Object-Oriented Programming:
- Classes define blueprints for objects
- __init__ method is the constructor
- self refers to the instance
- Inheritance: class Child(Parent):
- Polymorphism through method overriding

Important Libraries:
1. NumPy - Numerical computing
   - np.array() for efficient arrays
   - Broadcasting for element-wise operations
   - Linear algebra functions

2. Pandas - Data manipulation
   - DataFrames for tabular data
   - pd.read_csv(), pd.to_csv()
   - groupby() for aggregations

3. Matplotlib - Data visualization
   - plt.plot() for line plots
   - plt.scatter() for scatter plots
   - plt.subplots() for multiple plots

Best Practices:
- Use meaningful variable names
- Follow PEP 8 style guide
- Write docstrings for functions
- Use virtual environments
- Write unit tests
- Handle exceptions properly

Practice Projects:
1. Web scraper using requests and BeautifulSoup
2. Data analysis project with pandas
3. API using Flask or FastAPI
4. Machine learning model with scikit-learn

Resources:
- Python.org official documentation
- "Automate the Boring Stuff" book
- Codecademy Python course
- LeetCode for coding practice

Next Topics to Study:
- Decorators and context managers
- Async programming with asyncio
- Web frameworks (Django/Flask)
- Testing with pytest"""
    
    def _generate_ml_notes(self) -> str:
        return """Machine Learning Concepts Study Guide

Supervised Learning:
1. Classification
   - Predicts categories/classes
   - Examples: email spam detection, image recognition
   - Algorithms: Logistic Regression, Random Forest, SVM, Neural Networks
   - Evaluation: Accuracy, Precision, Recall, F1-score

2. Regression
   - Predicts continuous values
   - Examples: house prices, stock prices
   - Algorithms: Linear Regression, Polynomial Regression, Random Forest
   - Evaluation: MSE, MAE, R-squared

Unsupervised Learning:
1. Clustering
   - Groups similar data points
   - Examples: customer segmentation, gene sequencing
   - Algorithms: K-Means, DBSCAN, Hierarchical Clustering
   - Evaluation: Silhouette score, Elbow method

2. Dimensionality Reduction
   - Reduces feature space while preserving information
   - Examples: visualization, compression
   - Algorithms: PCA, t-SNE, UMAP

Data Preprocessing:
- Feature scaling: StandardScaler, MinMaxScaler
- Handling missing values: imputation, removal
- Encoding categorical variables: one-hot, label encoding
- Feature selection: correlation analysis, feature importance

Model Evaluation:
- Train/Validation/Test split
- Cross-validation
- Bias-variance tradeoff
- Overfitting vs. underfitting
- Learning curves

Key Algorithms:
1. Linear Regression
   - Simple, interpretable
   - Assumes linear relationship
   - Good baseline model

2. Random Forest
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Built-in feature importance

3. Neural Networks
   - Deep learning for complex patterns
   - Requires large datasets
   - Black box but powerful

Python Libraries:
- scikit-learn: General ML algorithms
- TensorFlow/PyTorch: Deep learning
- XGBoost: Gradient boosting
- pandas: Data manipulation
- matplotlib/seaborn: Visualization

Project Workflow:
1. Define problem and success metrics
2. Collect and explore data
3. Clean and preprocess data
4. Feature engineering
5. Model selection and training
6. Evaluation and validation
7. Deployment and monitoring

Next Steps:
- Complete Andrew Ng's ML course
- Work on Kaggle competitions
- Build end-to-end ML project
- Learn about MLOps and deployment"""
    
    def _generate_resources(self) -> str:
        return """Data Science Learning Resources

Online Courses:
1. Coursera - Andrew Ng's Machine Learning Course
   - Comprehensive introduction to ML
   - Mathematical foundations
   - Practical assignments in Octave/MATLAB

2. edX - MIT Introduction to Computer Science
   - Strong programming fundamentals
   - Problem-solving approach
   - Python-based

3. Udacity - Data Scientist Nanodegree
   - Project-based learning
   - Industry mentorship
   - Portfolio development

Books:
1. "Python for Data Analysis" by Wes McKinney
   - Pandas library deep dive
   - Data wrangling techniques
   - Real-world examples

2. "Hands-On Machine Learning" by Aurélien Géron
   - Practical ML implementation
   - Scikit-learn and TensorFlow
   - End-to-end projects

3. "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
   - Mathematical foundations
   - Advanced concepts
   - Reference book

YouTube Channels:
1. 3Blue1Brown - Mathematical intuition
2. StatQuest - Statistics explained simply
3. Two Minute Papers - Latest research summaries
4. Sentdex - Python tutorials

Websites and Blogs:
1. Kaggle Learn - Free micro-courses
2. Towards Data Science (Medium)
3. Analytics Vidhya
4. KDnuggets
5. Distill.pub - Visual explanations

Practice Platforms:
1. Kaggle - Competitions and datasets
2. LeetCode - Programming practice
3. HackerRank - Data science challenges
4. DataCamp - Interactive exercises

Tools to Learn:
1. Programming: Python, R, SQL
2. Visualization: Matplotlib, Seaborn, Plotly, Tableau
3. ML Libraries: Scikit-learn, TensorFlow, PyTorch
4. Big Data: Spark, Hadoop
5. Cloud: AWS, GCP, Azure

Project Ideas:
1. Stock price prediction
2. Customer churn analysis
3. Movie recommendation system
4. Sentiment analysis of tweets
5. Image classification
6. Sales forecasting

Communities:
1. Reddit: r/MachineLearning, r/datascience
2. Stack Overflow for technical questions
3. LinkedIn data science groups
4. Local meetups and conferences

Study Schedule:
- Monday: Theory and concepts
- Tuesday: Coding practice
- Wednesday: Project work
- Thursday: Reading research papers
- Friday: Review and practice problems
- Weekend: Long-form projects

Goals for 2024:
- Complete 3 online courses
- Read 5 data science books
- Finish 10 Kaggle competitions
- Build 5 portfolio projects
- Attend 2 data science conferences"""
    
    def _generate_certificate(self) -> str:
        return """Certificate of Completion

This certifies that

JOHN SMITH

has successfully completed the course

"Introduction to Machine Learning with Python"

Offered by DataCamp University
Course Duration: 40 hours
Completion Date: March 15, 2024

Course Content Covered:
- Python programming fundamentals
- NumPy and Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for machine learning
- Supervised learning algorithms
- Unsupervised learning techniques
- Model evaluation and validation
- Feature engineering
- Cross-validation techniques
- Ensemble methods

Skills Demonstrated:
✓ Data preprocessing and cleaning
✓ Exploratory data analysis
✓ Classification and regression modeling
✓ Model selection and hyperparameter tuning
✓ Performance evaluation and interpretation
✓ Visualization of results

Final Project: Customer Churn Prediction Model
- Achieved 87% accuracy on test set
- Used Random Forest and Logistic Regression
- Implemented feature engineering and selection
- Created comprehensive analysis report

Grade: A (92/100)

Instructor: Dr. Sarah Johnson, Ph.D.
Data Science Department
DataCamp University

Certificate ID: DC-ML-2024-JS-001
Verification: www.datacamp.edu/verify/DC-ML-2024-JS-001

This certificate demonstrates proficiency in fundamental machine learning concepts and practical implementation skills in Python."""


def main():
    """Main function to run demo data generation"""
    generator = TestDataGenerator()
    
    print("AI File Butler - Demo Data Generator")
    print("=" * 50)
    
    scenarios = {
        "1": ("Messy Downloads Folder", generator.create_messy_downloads_scenario),
        "2": ("Photo Collection", generator.create_photo_collection_scenario),
        "3": ("Document Archive", generator.create_document_archive_scenario),
        "4": ("All Scenarios", lambda: [
            generator.create_messy_downloads_scenario(),
            generator.create_photo_collection_scenario(),
            generator.create_document_archive_scenario()
        ])
    }
    
    print("Available test scenarios:")
    for key, (name, _) in scenarios.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect scenario to generate (1-4): ").strip()
    
    if choice in scenarios:
        name, func = scenarios[choice]
        print(f"\nGenerating: {name}")
        result = func()
        if isinstance(result, list):
            print(f"Created {len(result)} scenarios")
            for path in result:
                print(f"  - {path}")
        else:
            print(f"Created scenario at: {result}")
    else:
        print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()