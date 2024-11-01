# parenting-style-subreddits-analyser

## Research question

- **How do religious beliefs affect parenting styles?**

### Religions
- Christianity
- Islam
- Hinduism
- Atheism

### Operationalisation
- We measure parenting styles based on the semantics of parenting content in religious subreddits.
- We assume that if two religious subreddits have semantically similar parenting content, those religions have similar parenting styles. 

## Work split

### Avnee
- Literature review
- Making the presentation (with help from the team)

### Shreyansh
- Download data from Reddit
- Text cleaning:
  - Cleaned dataset ready in a DataFrame
  - Columns:
    - ID
    - Title
    - Text
    - Author
    - Created Time
    - URL
    - Score
    - Subreddit

### BC
- Filtering titles to extract parenting content from subreddits:
  - Use the title column from Shreyansh's Dataframe and an LLM
- Presenting the presentation

### Felix
- Embeddings
- Similarity metrics and visualisation

## Work conventions
- Work in branches
- Presentation on Google Drive
