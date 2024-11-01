# Importing the required libraries
import praw
import pytz
from datetime import datetime, timezone

class RedditScraper:

    def __init__(self, client_id, client_secret, user_agent):
        """
        Initializes RedditScraper with Reddit API credentials and user details.

        :param client_id: Reddit API client ID
        :param client_secret: Reddit API client secret
        :param user_agent: Reddit API user agent
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def scrape_subreddit(self, subreddit_name, limit=1000):
        """
        Scrapes posts from the specified subreddit and returns them as a dictionary.

        :param subreddit_name: The name of the subreddit to scrape
        :param limit: The number of posts to retrieve
        :return: A dictionary containing post details
        """

        data = {"ID":[],
                "title": [], 
                "author": [], 
                "created_time": [], 
                "url": [], 
                "score": [],
                "permalink": [], 
                "subreddit": [],
                "selftext": [],
                "num_comments": [], 
                "is_self": [], 
                "link_flair_text": [], 
                "link_flair_css_class": [], 
                "upvote_ratio": [], 
                "over_18_content": [], 
                "tag": []}

        subreddit = self.reddit.subreddit(subreddit_name)

        for post in subreddit.hot(limit=1000):
            
            id_ = post.id if post.id is not None else ""
            data["ID"].append(id_)
            
            
            title = post.title if post.title is not None else ""
            data["title"].append(title)

            author = post.author.name if post.author is not None else ""
            data["author"].append(author)

            created_utc = post.created_utc or 0
            utc_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            london_time = utc_time.astimezone(pytz.timezone('Europe/London'))
            data["created_time"].append(london_time)

            url = post.url if post.url is not None else ""
            data["url"].append(url)

            score = post.score if post.score is not None else ""
            data["score"].append(score)

            permalink = post.permalink if post.permalink is not None else ""
            data["permalink"].append(permalink)

            subreddit = post.subreddit.display_name if post.subreddit is not None else ""
            data["subreddit"].append(subreddit)

            selftext = post.selftext if post.selftext is not None else ""
            data["selftext"].append(selftext)

            num_comments = post.num_comments if post.num_comments is not None else ""
            data["num_comments"].append(num_comments)

            is_self = post.is_self if post.is_self is not None else ""
            data["is_self"].append(is_self)

            link_flair_text = post.link_flair_text if post.link_flair_text is not None else ""
            data["link_flair_text"].append(link_flair_text)

            link_flair_css_class = post.link_flair_css_class if post.link_flair_css_class is not None else ""
            data["link_flair_css_class"].append(link_flair_css_class)

            upvote_ratio = post.upvote_ratio if post.upvote_ratio is not None else ""
            data["upvote_ratio"].append(upvote_ratio)

            over_18 = post.over_18 if post.over_18 is not None else ""
            data["over_18_content"].append(over_18)

            tag = post.link_flair_text if post.link_flair_text is not None else ""
            data["tag"].append(tag)


        return data

