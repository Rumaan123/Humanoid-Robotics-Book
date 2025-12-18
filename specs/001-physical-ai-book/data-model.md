# Data Model: Physical AI & Humanoid Robotics Book

## Content Entities

### Module Content
- **Description**: Represents the structured information for each of the four official course modules
- **Attributes**:
  - moduleId: String (unique identifier for the module)
  - title: String (module title)
  - description: String (brief overview of the module)
  - topics: Array of Topic objects (list of topics within the module)
  - learningObjectives: Array of String (what readers should understand after the module)
  - prerequisites: Array of String (knowledge required before reading)
  - duration: Number (estimated reading time in minutes)
  - status: Enum (draft, review, approved)

### Topic
- **Description**: Represents a focused topic within a module
- **Attributes**:
  - topicId: String (unique identifier for the topic)
  - moduleId: String (reference to parent module)
  - title: String (topic title)
  - content: String (the actual content in Markdown format)
  - summary: String (brief summary of the topic)
  - keyConcepts: Array of String (important concepts covered)
  - relatedTopics: Array of String (references to related topics)
  - sources: Array of Source objects (citations for the topic)
  - difficulty: Enum (beginner, intermediate, advanced)
  - wordCount: Number (length of content)

### Source
- **Description**: Represents an authoritative source cited in the book
- **Attributes**:
  - sourceId: String (unique identifier)
  - title: String (title of the source)
  - authors: Array of String (authors of the source)
  - publication: String (journal, conference, or publisher)
  - year: Number (publication year)
  - url: String (URL to the source)
  - type: Enum (academic-paper, documentation, book, article, official-resource)
  - accessDate: Date (when the source was accessed for verification)
  - relevance: String (how the source relates to the topic)

### Concept
- **Description**: Represents a key concept that appears across multiple topics/modules
- **Attributes**:
  - conceptId: String (unique identifier)
  - name: String (name of the concept)
  - definition: String (clear definition of the concept)
  - modules: Array of String (modules where the concept appears)
  - topics: Array of String (topics where the concept is discussed)
  - relatedConcepts: Array of String (related concepts)
  - examples: Array of String (examples of the concept in practice)

### LearningObjective
- **Description**: Represents a specific learning outcome for readers
- **Attributes**:
  - objectiveId: String (unique identifier)
  - moduleId: String (module this objective belongs to)
  - description: String (what the reader should understand/know)
  - verificationMethod: String (how to verify the objective is met)
  - difficulty: Enum (beginner, intermediate, advanced)

## Content Relationships

### Module-Topic Relationship
- One Module contains many Topics
- Each Topic belongs to exactly one Module
- Topics are ordered within a Module to create learning progression

### Topic-Source Relationship
- One Topic references many Sources
- One Source may be referenced by many Topics
- This is a many-to-many relationship with citations

### Topic-Concept Relationship
- One Topic may discuss many Concepts
- One Concept may appear in many Topics
- This is a many-to-many relationship showing concept coverage

### Module-LearningObjective Relationship
- One Module has many LearningObjectives
- Each LearningObjective belongs to exactly one Module
- Objectives are aligned with module content

## Content Validation Rules

### Module Content Validation
- Each Module must have 1-10 Topics (prevent overly large or small modules)
- Each Module must have at least 3 LearningObjectives
- Module duration must be between 120-240 minutes (2-4 hours of reading)
- Module title must match one of the four official course modules

### Topic Validation
- Each Topic must have 500-2500 words (enough for depth but not overwhelming)
- Each Topic must reference at least 2 authoritative Sources
- Each Topic must be linked to at least 1 Concept
- Topic difficulty must align with module prerequisites

### Source Validation
- At least 50% of Sources must be authoritative or peer-reviewed (per constitution)
- Each Source must have a valid URL or DOI
- Source access date must be within 2 years for rapidly evolving technologies
- Sources must be properly formatted in APA style

### Concept Validation
- Each Concept must appear in at least 2 Topics (for consistency)
- Concept definitions must be consistent across all Topics
- Related concepts must form a coherent knowledge graph

## Content State Transitions

### Topic States
- Draft → Review (when content is complete and ready for review)
- Review → Approved (when reviewed and meets quality standards)
- Approved → Needs Update (when sources become outdated)

### Module States
- Draft → Review (when all topics in the module are approved)
- Review → Approved (when module meets overall quality standards)
- Approved → Complete (when all validation rules pass)

## Content Navigation Model

### Module Progression
- Modules must be read in sequence (Module 1 → Module 2 → Module 3 → Module 4)
- Prerequisites from earlier modules support later modules
- Cross-module references are allowed but should be minimal

### Topic Progression
- Topics within a module should be read in order
- Later topics may reference concepts from earlier topics in the same module
- Each topic should be self-contained enough for reference use