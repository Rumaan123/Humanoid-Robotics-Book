# Content Contracts: Physical AI & Humanoid Robotics Book

## Overview
This document defines the content contracts for the Physical AI & Humanoid Robotics book project. Since this is a documentation project, these contracts specify the interface expectations between different content components and the Docusaurus framework.

## Content Module Interface

### Module Metadata Contract
```
{
  "moduleId": "string (unique identifier)",
  "title": "string (module title)",
  "description": "string (brief overview)",
  "learningObjectives": "array of strings",
  "prerequisites": "array of strings",
  "estimatedReadingTime": "number (minutes)",
  "status": "enum (draft|review|approved|complete)",
  "topics": "array of topic references"
}
```

**Purpose**: Defines the standard metadata structure for each of the four course modules.

**Consumers**: Docusaurus navigation system, content management, reader progress tracking.

**Validation rules**:
- moduleId must be unique across all modules
- title must match one of the four official course modules
- estimatedReadingTime must be between 120-240 minutes
- status must be one of the defined enum values

## Topic Content Interface

### Topic Structure Contract
```
{
  "topicId": "string (unique identifier)",
  "moduleId": "string (module identifier)",
  "title": "string (topic title)",
  "content": "string (Markdown content)",
  "summary": "string (brief summary)",
  "keyConcepts": "array of concept references",
  "sources": "array of source references",
  "difficulty": "enum (beginner|intermediate|advanced)",
  "wordCount": "number (actual word count)"
}
```

**Purpose**: Defines the structure for individual topic content within modules.

**Consumers**: Docusaurus content rendering, search indexing, difficulty-based filtering.

**Validation rules**:
- topicId must be unique across all topics
- moduleId must reference an existing module
- content must be valid Markdown
- wordCount must be between 500-2500 words
- difficulty must align with module prerequisites

## Source Citation Interface

### Source Reference Contract
```
{
  "sourceId": "string (unique identifier)",
  "title": "string (source title)",
  "authors": "array of strings",
  "publication": "string (journal/publisher)",
  "year": "number (publication year)",
  "url": "string (valid URL)",
  "type": "enum (academic-paper|documentation|book|article|official-resource)",
  "accessDate": "string (ISO 8601 date)",
  "relevance": "string (explanation of relevance to topic)"
}
```

**Purpose**: Defines the standard format for source citations used throughout the book.

**Consumers**: Reference generation, bibliography compilation, source verification.

**Validation rules**:
- sourceId must be unique across all sources
- url must be a valid, accessible URL
- year must be a valid year (1900-current)
- type must be one of the defined enum values
- At least 50% of sources must be authoritative or peer-reviewed

## Concept Relationship Interface

### Concept Definition Contract
```
{
  "conceptId": "string (unique identifier)",
  "name": "string (concept name)",
  "definition": "string (clear definition)",
  "modules": "array of module references",
  "topics": "array of topic references",
  "relatedConcepts": "array of concept references",
  "examples": "array of example strings"
}
```

**Purpose**: Defines the structure for key concepts that appear across multiple topics.

**Consumers**: Concept mapping, cross-referencing, knowledge graph generation.

**Validation rules**:
- conceptId must be unique across all concepts
- definition must be clear and consistent
- Each concept must appear in at least 2 topics
- Related concepts must form valid references

## Navigation Contract

### Module Progression Contract
```
{
  "currentModule": "string (module identifier)",
  "prerequisites": "array of module identifiers",
  "nextModule": "string (module identifier or null)",
  "completionRequirements": {
    "topicsCompleted": "number",
    "totalTopics": "number",
    "minLearningObjectivesMet": "number"
  }
}
```

**Purpose**: Defines the navigation and progression logic between modules.

**Consumers**: Docusaurus navigation, learning path guidance, progress tracking.

**Validation rules**:
- Modules must be completed in sequence (1→2→3→4)
- Prerequisites must be completed before advancing
- At least 80% of learning objectives must be met before advancing

## Content Quality Contract

### Quality Verification Contract
```
{
  "contentId": "string (topic or module identifier)",
  "verificationChecks": {
    "technicalAccuracy": "boolean",
    "sourceVerification": "boolean",
    "citationFormat": "boolean",
    "readability": "boolean",
    "consistency": "boolean"
  },
  "verifier": "string (identity of reviewer)",
  "verificationDate": "string (ISO 8601 date)",
  "status": "enum (pending|verified|needs-revision)"
}
```

**Purpose**: Defines the quality verification process for all content.

**Consumers**: Review workflow, quality assurance, publication gatekeeping.

**Validation rules**:
- All verification checks must pass before status becomes "verified"
- Verification must be performed by qualified reviewer
- Status must be updated when content is modified

## Docusaurus Integration Contract

### Front Matter Schema
```
{
  "title": "string (page title)",
  "description": "string (page description)",
  "sidebar_label": "string (label for sidebar)",
  "sidebar_position": "number (position in sidebar)",
  "tags": "array of strings",
  "authors": "array of author references",
  "draft": "boolean (whether page is draft)"
}
```

**Purpose**: Defines the required front matter for Docusaurus compatibility.

**Consumers**: Docusaurus build process, search indexing, navigation generation.

**Validation rules**:
- All fields are required for proper Docusaurus integration
- sidebar_position must be unique within parent section
- draft pages are excluded from production builds