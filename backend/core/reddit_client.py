# open Reddit API client using PRAW
# fetch comments, posts, and search for keywords in subreddits:

# fetch_comments_from_subreddit(subreddit_name, limit=100, save_format='csv')
# fetch_hot_posts_from_subreddit(subreddit_name, limit=100, save_format='csv')
# search_subreddit(subreddit_name, keyword, time_filter='all', limit=100, save_format='csv')

import os
import sys
import praw
import pandas as pd
from datetime import datetime
import json
from dotenv import load_dotenv
from pathlib import Path

# Get the directory where this script is located (backend/core)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.insert(0, backend_dir)

from config import CLIENT_ID, SECRET_KEY, USERNAME, PASSWORD

if not all([CLIENT_ID, SECRET_KEY, USERNAME, PASSWORD]):
    raise ValueError("Missing Reddit credentials in .env file. Please check your .env file.")

# Create data directory structure inside backend folder
data_dir = os.path.join(backend_dir, 'data')
reddit_data_dir = os.path.join(data_dir, 'reddit')
raw_data_dir = os.path.join(reddit_data_dir, 'raw')
processed_data_dir = os.path.join(reddit_data_dir, 'processed')

# Create directories if they don't exist
for directory in [data_dir, reddit_data_dir, raw_data_dir, processed_data_dir]:
    os.makedirs(directory, exist_ok=True)

print(f"Backend directory: {backend_dir}")
print(f"Data will be saved to: {raw_data_dir}")

# Authentication
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=SECRET_KEY,
    user_agent='MyRedditBot/1.0 by Artysicist_Z',
    username=USERNAME,
    password=PASSWORD
)

try:
    print(f"Authenticated as: {reddit.user.me()}")
    print("✓ Authentication successful!")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    raise


def fetch_comments_from_subreddit(subreddit_name, limit=100, save_format='csv'):
    """
    Fetch comments from a subreddit and save to file.
    
    Args:
        subreddit_name: Name of the subreddit (e.g., 'wallstreetbets')
        limit: Number of comments to fetch
        save_format: 'csv', 'json', or 'both'
    
    Returns:
        DataFrame with comment data
    """
    print(f"\nFetching {limit} comments from r/{subreddit_name}...")
    
    subreddit = reddit.subreddit(subreddit_name)
    comments_data = []
    
    try:
        for comment in subreddit.comments(limit=limit):
            comment_info = {
                'comment_id': comment.id,
                'author': str(comment.author) if comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc),
                'submission_id': comment.submission.id,
                'submission_title': comment.submission.title,
                'subreddit': subreddit_name,
                'permalink': f"https://reddit.com{comment.permalink}",
                'is_submitter': comment.is_submitter,
                'edited': comment.edited if comment.edited else False,
                'controversiality': comment.controversiality,
                'gilded': comment.gilded
            }
            comments_data.append(comment_info)
        
        # Create DataFrame
        df = pd.DataFrame(comments_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{subreddit_name}_comments_{timestamp}"
        
        # Save based on format
        if save_format in ['csv', 'both']:
            csv_path = os.path.join(raw_data_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ Saved CSV to: {csv_path}")
        
        if save_format in ['json', 'both']:
            json_path = os.path.join(raw_data_dir, f"{base_filename}.json")
            df.to_json(json_path, orient='records', date_format='iso', indent=2)
            print(f"✓ Saved JSON to: {json_path}")
        
        print(f"✓ Fetched {len(df)} comments successfully!")
        return df
    
    except Exception as e:
        print(f"✗ Error fetching comments: {e}")
        return pd.DataFrame()


def fetch_submissions_from_subreddit(subreddit_name, sort_by='hot', time_filter='day', limit=100, save_format='csv'):
    """
    Fetch submissions (posts) from a subreddit.
    
    Args:
        subreddit_name: Name of the subreddit
        sort_by: 'hot', 'new', 'top', 'rising', 'controversial'
        time_filter: 'hour', 'day', 'week', 'month', 'year', 'all' (for 'top' and 'controversial')
        limit: Number of submissions to fetch
        save_format: 'csv', 'json', or 'both'
    
    Returns:
        DataFrame with submission data
    """
    print(f"\nFetching {limit} {sort_by} submissions from r/{subreddit_name}...")
    
    subreddit = reddit.subreddit(subreddit_name)
    submissions_data = []
    
    try:
        # Get submissions based on sort type
        if sort_by == 'hot':
            submissions = subreddit.hot(limit=limit)
        elif sort_by == 'new':
            submissions = subreddit.new(limit=limit)
        elif sort_by == 'top':
            submissions = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort_by == 'rising':
            submissions = subreddit.rising(limit=limit)
        elif sort_by == 'controversial':
            submissions = subreddit.controversial(time_filter=time_filter, limit=limit)
        else:
            submissions = subreddit.hot(limit=limit)
        
        for submission in submissions:
            submission_info = {
                'submission_id': submission.id,
                'title': submission.title,
                'author': str(submission.author) if submission.author else '[deleted]',
                'selftext': submission.selftext,
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'url': submission.url,
                'permalink': f"https://reddit.com{submission.permalink}",
                'subreddit': subreddit_name,
                'link_flair_text': submission.link_flair_text,
                'is_self': submission.is_self,
                'over_18': submission.over_18,
                'spoiler': submission.spoiler,
                'stickied': submission.stickied,
                'gilded': submission.gilded,
                'distinguished': submission.distinguished
            }
            submissions_data.append(submission_info)
        
        # Create DataFrame
        df = pd.DataFrame(submissions_data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{subreddit_name}_submissions_{sort_by}_{timestamp}"
        
        # Save
        if save_format in ['csv', 'both']:
            csv_path = os.path.join(raw_data_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ Saved CSV to: {csv_path}")
        
        if save_format in ['json', 'both']:
            json_path = os.path.join(raw_data_dir, f"{base_filename}.json")
            df.to_json(json_path, orient='records', date_format='iso', indent=2)
            print(f"✓ Saved JSON to: {json_path}")
        
        print(f"✓ Fetched {len(df)} submissions successfully!")
        return df
    
    except Exception as e:
        print(f"✗ Error fetching submissions: {e}")
        return pd.DataFrame()


def search_subreddit(subreddit_name, query, sort_by='relevance', time_filter='all', limit=100, save_format='csv'):
    """
    Search for posts in a subreddit containing specific keywords.
    
    Args:
        subreddit_name: Name of the subreddit
        query: Search query (e.g., 'NVDA', 'Tesla')
        sort_by: 'relevance', 'hot', 'top', 'new', 'comments'
        time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        limit: Number of results
        save_format: 'csv', 'json', or 'both'
    
    Returns:
        DataFrame with search results
    """
    print(f"\nSearching r/{subreddit_name} for '{query}'...")
    
    subreddit = reddit.subreddit(subreddit_name)
    search_data = []
    
    try:
        for submission in subreddit.search(query, sort=sort_by, time_filter=time_filter, limit=limit):
            submission_info = {
                'submission_id': submission.id,
                'title': submission.title,
                'author': str(submission.author) if submission.author else '[deleted]',
                'selftext': submission.selftext,
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'created_utc': datetime.fromtimestamp(submission.created_utc),
                'url': submission.url,
                'permalink': f"https://reddit.com{submission.permalink}",
                'search_query': query
            }
            search_data.append(submission_info)
        
        df = pd.DataFrame(search_data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_query = query.replace(' ', '_').replace('/', '_')
        base_filename = f"{subreddit_name}_search_{safe_query}_{timestamp}"
        
        # Save
        if save_format in ['csv', 'both']:
            csv_path = os.path.join(raw_data_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ Saved CSV to: {csv_path}")
        
        if save_format in ['json', 'both']:
            json_path = os.path.join(raw_data_dir, f"{base_filename}.json")
            df.to_json(json_path, orient='records', date_format='iso', indent=2)
            print(f"✓ Saved JSON to: {json_path}")
        
        print(f"✓ Found {len(df)} results!")
        return df
    
    except Exception as e:
        print(f"✗ Error searching: {e}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Fetch comments
    comments_df = fetch_comments_from_subreddit('wallstreetbets', limit=50, save_format='json')
    
    # Fetch hot submissions
    submissions_df = fetch_submissions_from_subreddit('stocks', sort_by='hot', limit=25, save_format='json')
    
    # Search for NVDA mentions
    nvda_df = search_subreddit('wallstreetbets', 'NVDA', time_filter='week', limit=30, save_format='json')

