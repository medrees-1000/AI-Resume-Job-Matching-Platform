"""
Enhanced job description cleaning and section extraction.
Better handles various job posting formats.
"""

import re

# Keywords that indicate relevant sections
RELEVANT_SECTION_KEYWORDS = [
    "responsibilities", "requirements", "qualifications", "required", "require",
    "preferred", "skills", "experience", "education", "what you'll do",
    "what you will do", "what you need", "you will", "must have",
    "should have", "nice to have", "technical skills", "key responsibilities",
    "core responsibilities", "essential", "desired", "minimum qualifications",
    "basic qualifications", "preferred qualifications", "technical requirements",
    "role requirements", "job requirements", "candidate profile"
]

# Keywords that indicate irrelevant sections  
IRRELEVANT_SECTION_KEYWORDS = [
    "about us", "about the company", "company overview", "who we are", "our mission", 
    "our values", "our culture", "why work here", "why join",
    "benefits", "compensation", "salary", "perks", "what we offer", "package",
    "equal opportunity", "eeo", "diversity", "application process", "apply now",
    "how to apply", "contact", "location details", "office location",
    "company description", "about the role", "team", "our team"
]

# Strong indicators of required vs preferred
REQUIRED_INDICATORS = [
    "required", "must have", "must-have", "essential", "mandatory", "minimum",
    "basic qualifications", "minimum qualifications", "requirements:",
    "required qualifications", "required skills", "you must", "you will need"
]

PREFERRED_INDICATORS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "plus", "desired",
    "ideal", "preferred qualifications", "preferred skills", "would be a plus",
    "it would be great if", "we'd love if", "additional", "optional"
]


def extract_requirements_section(job_text: str) -> dict:
    """
    Extract only the requirements/qualifications from job description.
    Enhanced to better separate required vs preferred skills.
    
    Returns:
        dict: {
            "cleaned_text": str (requirements only),
            "required_skills": str,
            "preferred_skills": str,
            "full_text": str (original)
        }
    """
    
    lines = job_text.split('\n')
    
    relevant_lines = []
    required_section = []
    preferred_section = []
    
    current_section = None
    skip_section = False
    in_relevant_section = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check if we should skip this section (company info, benefits, etc.)
        if any(keyword in line_lower for keyword in IRRELEVANT_SECTION_KEYWORDS):
            skip_section = True
            in_relevant_section = False
            continue
        
        # Check if this is a relevant section header
        if any(keyword in line_lower for keyword in RELEVANT_SECTION_KEYWORDS):
            skip_section = False
            in_relevant_section = True
            
            # Determine if it's required or preferred
            if any(word in line_lower for word in REQUIRED_INDICATORS):
                current_section = "required"
            elif any(word in line_lower for word in PREFERRED_INDICATORS):
                current_section = "preferred"
            else:
                # Ambiguous - might be either
                current_section = "general"
        
        # Add line if not skipping and in relevant section
        if not skip_section and (in_relevant_section or current_section):
            relevant_lines.append(line)
            
            # Add to appropriate section
            if current_section == "required":
                required_section.append(line)
            elif current_section == "preferred":
                preferred_section.append(line)
            elif current_section == "general":
                # If unclear, add to both (will be filtered later)
                required_section.append(line)
    
    # Combine sections
    cleaned_text = '\n'.join(relevant_lines)
    required_text = '\n'.join(required_section)
    preferred_text = '\n'.join(preferred_section)
    
    # If extraction failed, use heuristic
    if len(cleaned_text) < 100:
        cleaned_text = extract_middle_section(job_text)
        required_text = cleaned_text  # Use all as required if can't separate
    
    # Enhanced: Try regex-based separation as backup
    if not required_text and not preferred_text:
        sections = regex_separate_requirements(job_text)
        required_text = sections.get("required", cleaned_text)
        preferred_text = sections.get("preferred", "")
    
    return {
        "cleaned_text": cleaned_text,
        "required_skills": required_text,
        "preferred_skills": preferred_text,
        "full_text": job_text
    }


def extract_middle_section(text: str) -> str:
    """
    Fallback: Extract middle 60% of text (usually where requirements are).
    """
    lines = [l for l in text.split('\n') if l.strip()]
    
    if len(lines) < 10:
        return text
    
    # Skip first 20% and last 20%
    start_idx = len(lines) // 5
    end_idx = len(lines) - (len(lines) // 5)
    
    return '\n'.join(lines[start_idx:end_idx])


def regex_separate_requirements(job_text: str) -> dict:
    """
    Use regex to find and separate required vs preferred sections.
    More aggressive pattern matching.
    """
    
    # Try to find clear "Required" section
    required_patterns = [
        r'(?:required|must have|minimum|essential|basic qualifications?)[\s:]*\n([\s\S]*?)(?=\n(?:preferred|nice to have|bonus|desired|$))',
        r'(?:requirements?)[\s:]*\n([\s\S]*?)(?=\n(?:preferred|nice to have|bonus|$))',
        r'(?:minimum qualifications?)[\s:]*\n([\s\S]*?)(?=\n(?:preferred|desired|$))'
    ]
    
    required_text = ""
    for pattern in required_patterns:
        match = re.search(pattern, job_text, re.IGNORECASE | re.MULTILINE)
        if match:
            required_text = match.group(1).strip()
            break
    
    # Try to find clear "Preferred" section
    preferred_patterns = [
        r'(?:preferred|nice to have|bonus|desired|ideal)[\s:]*\n([\s\S]*?)(?=\n\n|\Z)',
        r'(?:preferred qualifications?)[\s:]*\n([\s\S]*?)(?=\n\n|\Z)',
        r'(?:it would be (?:great|nice|a plus) if)[\s\S]*?\n([\s\S]*?)(?=\n\n|\Z)'
    ]
    
    preferred_text = ""
    for pattern in preferred_patterns:
        match = re.search(pattern, job_text, re.IGNORECASE | re.MULTILINE)
        if match:
            preferred_text = match.group(1).strip()
            break
    
    return {
        "required": required_text,
        "preferred": preferred_text
    }


def smart_split_requirements(job_text: str) -> dict:
    """
    Intelligent splitting when explicit sections aren't found.
    Uses heuristics based on language patterns.
    """
    
    lines = job_text.split('\n')
    required_lines = []
    preferred_lines = []
    
    for line in lines:
        line_lower = line.lower()
        
        # Strong required indicators
        if any(indicator in line_lower for indicator in ["must have", "required", "must be", "essential"]):
            required_lines.append(line)
        # Strong preferred indicators
        elif any(indicator in line_lower for indicator in ["preferred", "nice to have", "bonus", "plus"]):
            preferred_lines.append(line)
        # Bullet points with numbers often indicate years of experience (required)
        elif re.search(r'\d+\+?\s*years?', line_lower):
            required_lines.append(line)
        # Ambiguous - default to required
        else:
            required_lines.append(line)
    
    return {
        "required": '\n'.join(required_lines),
        "preferred": '\n'.join(preferred_lines)
    }