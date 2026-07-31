# EMBER Website Build Brief

## Project Overview
Create a public-facing website for **EMBER**, a wildfire mitigation and planning platform. The site should communicate what EMBER does, show mitigation plans and efforts, explain risk and preparedness concepts in a clear way, and present EMBER as a modern, intelligent, trustworthy system.

The website should feel **futuristic, beautiful, intuitive, and professional**. It should also be inspired by the **Rothermel model** in the sense that it should visually and conceptually communicate wildfire behavior, spread logic, mitigation planning, and decision support. The design does not need to literally recreate a scientific model interface, but it should borrow from that feeling: data-informed, spatial, layered, analytical, and visually striking.

The build should be **containerized using Docker**, with a **FastAPI backend** and a **React + Vite frontend**.

---

## Core Goals
1. Explain what EMBER is in a way that is accessible to the public, researchers, and stakeholders.
2. Show mitigation plans, preparedness strategies, and wildfire risk reduction efforts.
3. Demonstrate the broader capabilities of EMBER beyond static information.
4. Create a polished platform identity that feels credible, futuristic, and easy to use.
5. Build the project with a scalable full-stack architecture that can grow into a richer decision-support system.

---

## Brand / Design Direction
The site should feel:
- Futuristic
- Clean
- Spatial
- Elegant
- Data-driven
- Calm but powerful
- High-tech without feeling cluttered
- Easy for nontechnical users to understand

### Visual Inspiration
The design should take inspiration from:
- Fire spread simulation visuals
- Heat maps
- Terrain overlays
- Layered environmental data
- Minimal dashboards
- Elegant dark-mode interfaces
- Strategic use of glowing accents

### Suggested Style
- Dark background with warm accent colors like ember orange, soft red, amber, copper, or gold
- Cool supporting tones like charcoal, slate, deep navy, and muted blue-gray
- Glassmorphism or subtle translucent panels
- Soft gradients that suggest heat, smoke, atmosphere, and terrain
- Motion effects that feel smooth and intelligent, not flashy
- Rounded corners, layered cards, and strong whitespace
- Map-inspired visual elements and subtle grid systems

### Typography
Use modern, clean sans-serif typography. Suggested vibe:
- Strong, bold headings
- Highly readable body text
- Clear hierarchy
- Spacious layout

---

## Target Audience
The website should be understandable and useful for:
- General public users
- Communities in wildfire-prone areas
- Local governments and planners
- Researchers and students
- Emergency management stakeholders
- Potential collaborators or funders

---

## Site Structure

## 1. Home Page
The homepage should immediately explain what EMBER is and why it matters.

### Hero Section
Include:
- A bold headline
- A short explanation of EMBER
- A strong visual background inspired by wildfire spread, terrain, heat signatures, or mitigation planning
- Clear call-to-action buttons such as:
  - Explore Mitigation Plans
  - View Risk Insights
  - Learn How EMBER Works

### Suggested Hero Messaging
**EMBER helps communities understand wildfire risk and act before disaster happens.**

Possible supporting text:
EMBER is a wildfire mitigation and decision-support platform designed to help communities explore risk, view mitigation strategies, and understand actions that can reduce wildfire impacts before a fire begins.

### Homepage Sections
- What EMBER does
- Why mitigation matters
- Key capabilities
- Featured mitigation plans or community examples
- Interactive overview of wildfire risk and preparedness
- A visual explanation of how EMBER works
- Call to action for exploring tools or contacting the team

---

## 2. About EMBER
This page should explain:
- What EMBER is
- Why it was created
- The problem it solves
- The science and systems-thinking approach behind it
- How it supports mitigation, planning, and preparedness

Suggested subsections:
- Mission
- Vision
- Why wildfire mitigation matters
- How EMBER is different
- Data-informed planning philosophy

---

## 3. Mitigation Plans
This should be one of the main sections of the site.

### What it should include
- Community mitigation plans
- Regional mitigation strategies
- Fuel treatment recommendations
- Preparedness guidelines
- Infrastructure and defensible space planning
- Prevention priorities by risk type

### Nice features
- Cards or map-based browsing
- Filter by region, plan type, hazard level, or category
- Search bar
- Download or view detailed plan pages
- Tags such as:
  - Fuel Management
  - Defensible Space
  - Monitoring
  - Prescribed Fire
  - Community Planning
  - Resource Allocation

---

## 4. Risk Insights / Data Explorer
This section should visually communicate wildfire risk patterns and supporting intelligence.

Possible content:
- Seasonal wildfire trends
- High-risk periods
- Geographic hotspots
- Cause-based patterns such as human-caused vs lightning-caused
- Preparedness indicators
- Weather-linked risk summaries
- Spatial clustering visuals

### Suggested UI
- Interactive map
- Layer toggles
- Charts and visual summaries
- Cards explaining what the data means in plain language

This section should feel analytical but still accessible.

---

## 5. What EMBER Can Do
This is where you explain all the system capabilities beyond just showing plans.

Possible capability categories:
- Show mitigation recommendations
- Organize and surface wildfire planning documents
- Explain risk patterns in plain language
- Support community preparedness education
- Provide region-specific insights
- Highlight vulnerable areas and infrastructure concerns
- Serve as a knowledge hub for mitigation efforts
- Potential future AI assistant or RAG-powered planning guidance
- Potential integration with maps, weather data, and mitigation databases

This page should make EMBER feel like a platform, not just a static website.

---

## 6. How It Works
This page should explain the system architecture and reasoning in a simple and visual way.

Suggested content:
- Data sources
- Risk modeling concepts
- Mitigation planning logic
- Document retrieval / knowledge system
- Visualization tools
- Future predictive or interactive features

This page can include a simplified system diagram.

---

## 7. Resources / Library
A searchable knowledge center for:
- Wildfire mitigation PDFs
- Preparedness guides
- Planning resources
- Educational documents
- Research summaries
- Downloadable tools or templates

This would be a great place to eventually connect a RAG or GRAG system.

---

## 8. Contact / Collaborate
A simple page for:
- Contact form
- Project inquiries
- Research collaboration
- Community partnership requests
- Demo requests

---

## Recommended Features

### Must-Have Features
- Responsive design for desktop, tablet, and mobile
- Smooth navigation
- Searchable mitigation plan library
- Clean information hierarchy
- Beautiful interactive homepage
- Clear calls to action
- Accessible design
- Fast page load times

### High-Impact Features
- Interactive map for mitigation plans and risk zones
- Timeline or seasonal preparedness view
- Search + filter system
- Animated wildfire spread-inspired visuals
- Layered cards with concise summaries
- Download center for PDFs and reports
- Visual explanation panels
- Smart glossary for wildfire terms

### Future-Ready Features
- User accounts for agencies or communities
- Saved mitigation plan dashboards
- AI assistant for mitigation guidance
- RAG/GRAG document question answering
- Risk score calculator
- Weather integration
- Geospatial overlays
- Scenario comparison tools
- Community-specific action recommendations

---

## Architecture Requirements

## Frontend
Use:
- React
- Vite
- TypeScript preferred
- Tailwind CSS for styling
- Component-based architecture
- Framer Motion for subtle animations
- React Router for navigation

### Frontend Suggestions
- Use a sleek design system
- Create reusable card, map panel, hero, section header, and stat components
- Prioritize smooth page transitions and hover effects
- Keep layout modular and scalable

---

## Backend
Use:
- FastAPI
- Python
- REST API structure
- Clear modular service organization

### Backend Responsibilities
- Serve mitigation plans and metadata
- Serve risk insight data
- Power search and filtering
- Handle contact form submissions
- Support document indexing or future RAG endpoints
- Support future user and admin features

### Suggested Backend Structure
- `api/routes`
- `services`
- `models`
- `schemas`
- `core`
- `utils`

Possible endpoints:
- `/plans`
- `/plans/{id}`
- `/risk`
- `/capabilities`
- `/resources`
- `/contact`
- `/search`

---

## Containerization
The project should be fully containerized with Docker.

### Requirements
- Separate frontend and backend services
- Dockerfiles for both frontend and backend
- Docker Compose for local orchestration
- Environment variable support
- Clean developer startup workflow

### Suggested Services
- `frontend`
- `backend`
- optional future services:
  - `db`
  - `vectorstore`
  - `nginx`

---

## Suggested Tech Stack
### Frontend
- React
- Vite
- TypeScript
- Tailwind CSS
- Framer Motion
- React Router

### Backend
- FastAPI
- Pydantic
- Uvicorn

### Nice Additions
- PostgreSQL for future structured data
- Mapbox or Leaflet for map interaction
- OpenAI or another model provider for future assistant features
- Vector database later if you add RAG or GRAG
- Nginx for production routing

---

## Design Suggestions That Would Make the Site Really Cool
1. **Interactive wildfire-inspired hero**
   - Animated glowing terrain lines
   - Heat-map-like motion
   - Floating stat cards

2. **Map-first browsing**
   - Let users explore mitigation plans geographically

3. **Layered data aesthetic**
   - Panels that feel like wildfire intelligence overlays

4. **Explainers in plain language**
   - Every data-heavy visual should also have a short human-readable explanation

5. **Beautiful microinteractions**
   - Smooth hover states
   - Fading transitions
   - Smart loading skeletons

6. **“Before the fire” framing**
   - Keep the message centered on prevention, preparedness, and mitigation

7. **Clean storytelling**
   - The site should tell a story:
   risk exists → mitigation is possible → EMBER helps guide action

---

## Content Suggestions
Include language around:
- Prevention before disaster
- Community resilience
- Data-informed mitigation
- Spatial risk awareness
- Preparedness and planning
- Smarter wildfire decision support

Potential homepage keywords:
- Mitigation
- Preparedness
- Risk
- Resilience
- Planning
- Community protection
- Decision support
- Wildfire intelligence

---

## Suggested Copilot Build Prompt
Build a full-stack website called **EMBER** with a **React + Vite frontend** and a **FastAPI backend**, containerized with **Docker**. The site should be futuristic, elegant, easy to use, and visually inspired by wildfire spread modeling and spatial risk systems similar in spirit to the Rothermel model. Use a dark, modern interface with ember-colored highlights, glassy layered cards, subtle motion, and a clean information hierarchy. The website should include pages for Home, About, Mitigation Plans, Risk Insights, What EMBER Can Do, How It Works, Resources, and Contact. The frontend should use reusable components, Tailwind CSS, Framer Motion, and React Router. The backend should expose endpoints for mitigation plans, risk data, resources, search, and contact. The project should include Dockerfiles for frontend and backend plus a docker-compose setup. Design the site so it feels credible, public-facing, and expandable into a future RAG/GRAG wildfire mitigation assistant.

---

## Final Notes
This website should not feel like a generic corporate template. It should feel like a modern wildfire intelligence platform that is visually compelling, strategically designed, and ready to grow into something bigger.

The most important balance is:
- beautiful but not cluttered
- futuristic but not confusing
- data-rich but still human-friendly
- technically credible but accessible

EMBER should feel like a platform that helps people act early, understand risk clearly, and trust the mitigation guidance they are seeing.
