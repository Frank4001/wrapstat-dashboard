# WrapStat Training Survey Data Dictionary

## Overview
This data dictionary describes the cleaned datasets created from the Illinois CCSO WrapStat Super User Training survey responses.

## Files Created

### 1. `cleaned_wrapstat_training_data.csv`
- **Purpose**: Clean, structured version of the original data with proper headers
- **Rows**: 38 survey responses
- **Columns**: 33 fields

### 2. `enhanced_wrapstat_training_data.csv` 
- **Purpose**: Full analysis-ready dataset with calculated fields and numeric mappings
- **Rows**: 38 survey responses  
- **Columns**: 60+ fields (includes derived metrics)

### 3. `dashboard_ready_wrapstat_data.csv`
- **Purpose**: Simplified dataset optimized for dashboard creation
- **Rows**: 38 survey responses
- **Columns**: 17 key fields

## Column Definitions

### Basic Information
| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `start_date` | When participant started training | DateTime | 2024-11-08 16:37:20 |
| `end_date` | When participant finished training | DateTime | 2024-11-08 16:38:46 |
| `recorded_date` | When response was recorded | DateTime | 2024-11-08 16:38:47 |
| `duration_seconds` | Training duration in seconds | Integer | 86 |
| `training_duration_minutes` | Training duration in minutes | Float | 1.4 |
| `response_id` | Unique survey response identifier | String | R_6PwuFEecQRNc7mh |
| `progress_pct` | Completion percentage | Integer | 100 |
| `completed` | Whether training was completed | Boolean | True |

### Geographic Information
| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `latitude` | Geographic latitude | Float | 38.5889 |
| `longitude` | Geographic longitude | Float | -89.9904 |
| `region` | Illinois region (derived) | String | Central Illinois |

### Knowledge Assessment
| Column | Description | Data Type | Scale |
|--------|-------------|-----------|-------|
| `knowledge_before` | Self-assessed knowledge before training | String | Extremely bad → Extremely good |
| `knowledge_after` | Self-assessed knowledge after training | String | Extremely bad → Extremely good |
| `knowledge_before_numeric` | Numeric version (1-5 scale) | Integer | 1=Extremely bad, 5=Extremely good |
| `knowledge_after_numeric` | Numeric version (1-5 scale) | Integer | 1=Extremely bad, 5=Extremely good |
| `knowledge_improvement` | Change in knowledge score | Integer | -4 to +4 |

### Training Effectiveness Ratings
All measured on 5-point Likert scale: Strongly disagree (1) → Strongly agree (5)

| Column | Description |
|--------|-------------|
| `improved_user_access` | Course improved ability to manage WrapStat user access |
| `improved_care_coordinator_mgmt` | Course improved ability to add/update Care Coordinators |
| `improved_youth_roster` | Course improved ability to manage Youth Roster |
| `improved_followup` | Course improved ability to follow up with non-respondents |

### Content Quality Ratings
All measured on 5-point Likert scale: Strongly disagree (1) → Strongly agree (5)

| Column | Description |
|--------|-------------|
| `content_engaging` | Content was engaging/held attention |
| `content_relevant` | Content was relevant to job |
| `content_understandable` | Content was easy to understand |
| `content_interactive` | Content was appropriately interactive |
| `content_visual_support` | Content was supported by visuals |
| `content_user_friendly` | Content was user-friendly |

### Technology Experience Ratings  
All measured on 5-point Likert scale: Strongly disagree (1) → Strongly agree (5)

| Column | Description |
|--------|-------------|
| `tech_easy_access` | Technology was easy to access |
| `tech_easy_navigate` | Technology was easy to navigate |
| `tech_functioning_properly` | Technology was functioning properly |

### Satisfaction Metrics
| Column | Description | Data Type | Range |
|--------|-------------|-----------|-------|
| `would_take_another_course` | Would take another course in this format | String | Yes/No (Please Explain) |
| `would_recommend` | Binary recommendation (derived) | Integer | 0=No, 1=Yes |
| `had_technical_difficulties` | Encountered technical difficulties | String | Yes/No (Please Explain) |
| `had_tech_issues` | Binary tech issues flag (derived) | Integer | 0=No, 1=Yes |

### Calculated Satisfaction Scores
| Column | Description | Data Type | Range |
|--------|-------------|-----------|-------|
| `avg_content_satisfaction` | Average of all content rating questions | Float | 1.0-5.0 |
| `avg_tech_satisfaction` | Average of all technology rating questions | Float | 1.0-5.0 |
| `avg_improvement_rating` | Average of all improvement questions | Float | 1.0-5.0 |

### Feedback
| Column | Description | Data Type |
|--------|-------------|-----------|
| `final_comments` | Open-ended feedback | String |
| `provided_comments` | Whether participant provided comments | Integer |
| `would_take_another_course_reason` | Explanation if wouldn't take another course | String |
| `technical_difficulties_explanation` | Details of technical difficulties | String |

### Time-based Analysis Fields
| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| `training_month` | Year-Month of training | String | 2024-11 |
| `training_year` | Year of training | Integer | 2024 |

## Key Insights from Data Summary

- **Response Rate**: 38 total responses
- **Time Period**: November 2024 - June 2025  
- **Completion Rate**: 100% (all participants completed training)
- **Average Duration**: 1.7 minutes
- **Recommendation Rate**: 86.8% would take another course
- **Knowledge Improvement**: Average increase of 1.29 points (on 5-point scale)
- **Technical Issues**: Only 5.3% experienced technical difficulties
- **Content Satisfaction**: 4.04/5.0 average rating
- **Technology Satisfaction**: 4.25/5.0 average rating

## Dashboard Recommendations

### Key Metrics to Display
1. **Overall Satisfaction**: Recommendation rate, average ratings
2. **Learning Effectiveness**: Knowledge improvement scores
3. **Content Quality**: Content satisfaction by dimension
4. **Technology Performance**: Tech satisfaction and issue rates
5. **Geographic Distribution**: Response patterns by Illinois region
6. **Trends Over Time**: Monthly response patterns and ratings

### Suggested Visualizations
- **Bar Charts**: Satisfaction ratings, knowledge before/after
- **Line Charts**: Trends over time
- **Maps**: Geographic distribution of participants
- **Word Clouds**: Common themes in open-ended feedback
- **Histograms**: Training duration distribution

### Filter Options
- Date range
- Geographic region  
- Knowledge improvement level
- Technical difficulty status
- Recommendation status