import pandas as pd
import numpy as np
from bisect import bisect_left
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
from collections import deque
import heapq
from utils.trie import Trie

# 감정과 장르 매핑
emotion_genre_map = {
    'Joy': ['코미디', '드라마'],
    'Sadness': ['드라마', '멜로/로맨스'],
    'Anger': ['액션', '서스펜스', '느와르'],
    'Disgust': ['공포', '미스터리'],
    'Fear': ['다큐멘터리', '애니메이션']
}

# 이진 탐색을 이용한 장르 필터링
def binary_search_genres(genres, target_genre):
    index = bisect_left(genres, target_genre)
    if index < len(genres) and genres[index] == target_genre:
        return True
    return False

def filter_movies_by_genre(movie_data, genres):
    genres.sort()  # 이진 탐색을 위해 정렬
    filtered_movies = movie_data[movie_data['genre'].apply(lambda x: any(binary_search_genres(genres, genre) for genre in x.split('|')))]
    return filtered_movies

# 가중치 기반 추천 점수 계산
def calculate_movie_scores(filtered_movies):
    filtered_movies['score'] = filtered_movies.apply(lambda row: 
                                                     (row['box_off_num'] * 0.4) + 
                                                     (row['num_staff'] * 0.1) + 
                                                     (row['num_actor'] * 0.1) + 
                                                     (row['time'] * 0.1) + 
                                                     (row['dir_prev_num'] * 0.2), axis=1)
    return filtered_movies

# 콘텐츠 기반 필터링을 위한 유사도 점수 계산
def calculate_similarity_scores(filtered_movies):
    # 장르와 감독을 결합한 텍스트 데이터 생성
    combined_features = filtered_movies['genre'] + " " + filtered_movies['director']
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(stop_words=None)
    tfidf_matrix = tfidf.fit_transform(combined_features)
    
    # 코사인 유사도 계산
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# 유사도 점수와 기존 점수를 결합하여 최종 점수 계산
def integrate_scores(filtered_movies, cosine_sim):
    similarity_score = cosine_sim.sum(axis=1)
    filtered_movies['integrated_score'] = filtered_movies['score'] * 0.5 + similarity_score * 0.5
    return filtered_movies

# 해시맵을 이용한 영화 데이터 검색
def create_movie_hashmap(movie_data):
    movie_hashmap = {}
    for _, row in movie_data.iterrows():
        movie_hashmap[row['title']] = row
    return movie_hashmap

# Trie를 이용한 영화 제목 검색
def create_movie_trie(movie_data):
    movie_trie = Trie()
    for title in movie_data['title']:
        movie_trie.insert(title)
    return movie_trie

def search_movies_by_prefix(movie_trie, prefix):
    return movie_trie.starts_with(prefix)

# 우선순위 큐를 이용한 영화 정렬 및 추천
def recommend_movie(emotion, movie_data, emotion_genre_map, recent_recommendations, movie_hashmap):
    if emotion not in emotion_genre_map:
        return None

    genres = emotion_genre_map[emotion]
    filtered_movies = filter_movies_by_genre(movie_data, genres)
    
    if filtered_movies.empty:
        return None

    filtered_movies = calculate_movie_scores(filtered_movies)
    cosine_sim = calculate_similarity_scores(filtered_movies)
    filtered_movies = integrate_scores(filtered_movies, cosine_sim)
    
    movies_list = filtered_movies.to_dict('records')
    
    # 우선순위 큐 사용
    movie_heap = []
    for movie in movies_list:
        if movie['title'] not in recent_recommendations:
            heapq.heappush(movie_heap, (-movie['integrated_score'], movie))  # 점수가 높은 순서대로 정렬

    # 무작위 요소를 추가하여 상위 영화 중 일부를 선택
    top_movies = [heapq.heappop(movie_heap)[1] for _ in range(min(10, len(movie_heap)))]
    top_movies = random.sample(top_movies, min(3, len(top_movies)))

    recommendations = [(movie['title'], movie['release_time']) for movie in top_movies]
    return recommendations

def recommend(emotion, movie_data, recent_recommendations, movie_hashmap, max_recent=20):
    recommendations = recommend_movie(emotion, movie_data, emotion_genre_map, recent_recommendations, movie_hashmap)
    if recommendations:
        # 최근 추천된 영화 목록 업데이트
        for title, release_time in recommendations:
            if len(recent_recommendations) >= max_recent:
                recent_recommendations.popleft()  # 가장 오래된 항목 제거
            recent_recommendations.append(title)
        return [(title, release_time.split('-')[0]) for title, release_time in recommendations]
    return None

def load_movie_data(filepath):
    movie_data = pd.read_csv(filepath)
    movie_hashmap = create_movie_hashmap(movie_data)
    movie_trie = create_movie_trie(movie_data)
    return movie_data, movie_hashmap, movie_trie

