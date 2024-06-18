import pandas as pd
from collections import deque
from movie_recommendation import recommend, recommend_movie, emotion_genre_map, create_movie_hashmap

def test_recommendation_system():
    # 실제 예시 데이터
    data = {
        'title': ['녹색의자 2013 - 러브 컨셉츄얼리', '사사건건', '국제시장', '앵두야, 연애하자', '비치하트애솔', 
                  '행복한 울릉인', '뭘 또 그렇게까지', '내부자들: 디 오리지널', '열한시', '미확인 동영상 : 절대클릭금지'],
        'distributor': ['(주)마운틴픽쳐스', 'KT&G 상상마당', 'CJ 엔터테인먼트', '홀리가든', '골든타이드픽처스', 
                        '마운틴 픽처스', '스폰지', '(주)쇼박스', 'CJ 엔터테인먼트', '(주)쇼박스'],
        'genre': ['멜로/로맨스', '드라마', '드라마', '드라마', '드라마', '다큐멘터리', '멜로/로맨스', '느와르', '공포', '공포'],
        'release_time': ['2013-10-31', '2010-01-21', '2014-12-17', '2013-06-06', '2015-01-29', 
                         '2010-02-25', '2010-04-29', '2015-12-31', '2013-11-28', '2012-05-30'],
        'time': [97, 91, 126, 98, 100, 76, 78, 180, 99, 93],
        'screening_rat': ['청소년 관람불가', '청소년 관람불가', '12세 관람가', '15세 관람가', '청소년 관람불가', '전체 관람가', '12세 관람가', '청소년 관람불가', '15세 관람가', '15세 관람가'],
        'director': ['박철수', '이정욱', '윤제균', '정하린', '이난', '황석호', '전계수', '우민호', '김현석', '김태경'],
        'dir_prev_bfnum': [0, 0, 0, 0, 0, 0, 0, 3, 2, 1],
        'dir_prev_num': [0, 0, 0, 0, 0, 0, 0, 3, 2, 1],
        'num_staff': [42, 148, 869, 34, 101, 3, 69, 382, 277, 197],
        'num_actor': [2, 8, 4, 4, 2, 2, 3, 3, 3, 3],
        'box_off_num': [366, 1143, 14262766, 3177, 279, 1068, 409, 2084844, 870785, 867386]
    }
    
    movie_data = pd.DataFrame(data)

    # 감정 키워드 테스트
    emotions = ['Joy', 'Sadness', 'Anger', 'Disgust', 'Fear']
    for emotion in emotions:
        print(f"\nRecommendations for emotion: {emotion}")
        recommendations = recommend_movie(emotion, movie_data, emotion_genre_map, deque(), create_movie_hashmap(movie_data))
        if recommendations:
            print("Filtered movies by genre:")
            filtered_movies = movie_data[movie_data['genre'].apply(lambda x: any(genre in x for genre in emotion_genre_map[emotion]))]
            print(filtered_movies[['title', 'genre']])
            print("Movies with scores:")
            filtered_movies['score'] = filtered_movies.apply(lambda row: 
                                                             (row['box_off_num'] * 0.4) + 
                                                             (row['num_staff'] * 0.1) + 
                                                             (row['num_actor'] * 0.1) + 
                                                             (row['time'] * 0.1) + 
                                                             (row['dir_prev_num'] * 0.2), axis=1)
            print(filtered_movies[['title', 'score']])
            print("Top recommended movies:")
            top_movies = filtered_movies.sort_values(by='score', ascending=False).head(3)
            print(top_movies[['title', 'score']])
            print("Final recommendations:", [f"{title} ({release_time.split('-')[0]})" for title, release_time in recommendations])
        else:
            print("No recommendations available.")

# 테스트 함수 실행
if __name__ == "__main__":
    test_recommendation_system()
