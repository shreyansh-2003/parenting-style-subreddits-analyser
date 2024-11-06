import praw
import pytz
from datetime import datetime, timezone, timedelta
import time
import pandas as pd
from typing import Dict, List
import logging

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initializes an instance of the RedditScraper class with Reddit API credentials.

        Args:
            client_id (str): The client ID for Reddit API authentication.
            client_secret (str): The client secret for Reddit API authentication.
            user_agent (str): The user agent string identifying the application.
        """
        # Initialize Reddit API connection
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Configure logging for information and error messages
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Template for storing data extracted from posts
        self.data_template = {
            "ID": [], "title": [], "author": [], "created_time": [],
            "url": [], "score": [], "permalink": [], "subreddit": [],
            "selftext": [], "num_comments": [], "is_self": [],
            "link_flair_text": [], "link_flair_css_class": [],
            "upvote_ratio": [], "over_18_content": [], "tag": []
        }

    def _extract_post_data(self, post) -> Dict:
        """
        Extracts relevant information from a single Reddit post.

        Args:
            post (praw.models.Submission): The Reddit post to extract data from.

        Returns:
            dict: A dictionary containing key data fields of the post, or None if an error occurs.
        """
        try:
            created_utc = post.created_utc or 0
            utc_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            london_time = utc_time.astimezone(pytz.timezone('Europe/London'))

            return {
                "ID": post.id or "",
                "title": post.title or "",
                "author": post.author.name if post.author else "[deleted]",
                "created_time": london_time,
                "url": post.url or "",
                "score": post.score or 0,
                "permalink": post.permalink or "",
                "subreddit": post.subreddit.display_name or "",
                "selftext": post.selftext or "",
                "num_comments": post.num_comments or 0,
                "is_self": post.is_self or False,
                "link_flair_text": post.link_flair_text or "",
                "link_flair_css_class": post.link_flair_css_class or "",
                "upvote_ratio": post.upvote_ratio or 0.0,
                "over_18_content": post.over_18 or False,
                "tag": post.link_flair_text or ""
            }
        except Exception as e:
            self.logger.error(f"Error extracting data from post {post.id}: {str(e)}")
            return None

    def _process_posts(self, posts_iterator, data: Dict, min_score: int) -> int:
        """
        Iterates over posts and adds qualifying data to the specified dictionary.

        Args:
            posts_iterator (generator): An iterator containing Reddit posts.
            data (dict): Dictionary to populate with extracted post data.
            min_score (int): Minimum score threshold for posts to include.

        Returns:
            int: The number of posts added to the data dictionary.
        """
        posts_added = 0
        for post in posts_iterator:
            try:
                # Only include posts meeting the minimum score requirement
                if min_score is not None and post.score < min_score:
                    continue

                # Extract and append post data if extraction is successful
                post_data = self._extract_post_data(post)
                if post_data:
                    for key, value in post_data.items():
                        data[key].append(value)
                    posts_added += 1
                    
                # Log progress every 100 posts
                if posts_added % 100 == 0:
                    self.logger.info(f"Processed {posts_added} posts")
                    
            except Exception as e:
                self.logger.error(f"Error processing post: {str(e)}")
                continue
                
        return posts_added

    def scrape_subreddit(self, 
                        subreddit_name: str,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        time_window_days: int = 30,
                        min_score: int = None,
                        max_retries: int = 3,
                        sleep_time: int = 2) -> Dict:
        """
        Scrapes posts from a specified subreddit within a given date range.

        Args:
            subreddit_name (str): The name of the subreddit to scrape.
            start_date (datetime): Start date for scraping posts.
            end_date (datetime): End date for scraping posts.
            time_window_days (int): Interval in days for each scraping session.
            min_score (int): Minimum score threshold for posts to include.
            max_retries (int): Maximum number of retries for network requests.
            sleep_time (int): Pause duration between requests in seconds.

        Returns:
            dict: A dictionary containing the scraped post data.
        """
        
        data = self.data_template.copy()
        subreddit = self.reddit.subreddit(subreddit_name)
        total_posts = 0
        
        # Set default start and end dates if not provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=365)

        self.logger.info(f"Starting scrape for r/{subreddit_name} from {start_date} to {end_date}")

        # Scrape top posts using different time filters for broader data coverage
        self.logger.info("Attempting to scrape using the subreddit.top() method.")
        for time_filter in ['all', 'year', 'month']:
            try:
                posts = subreddit.top(time_filter=time_filter, limit=None)
                posts_added = self._process_posts(posts, data, min_score)
                total_posts += posts_added
                self.logger.info(f"Added {posts_added} posts using top/{time_filter}")
                time.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"Error fetching top posts for {time_filter}: {str(e)}")

        return data
