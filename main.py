import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_preprocessing import preprocess_data, load_data
from model import BERTClassifier, BERTDataset, initialize_model
from collections import deque
from movie_recommendation import load_movie_data, search_movies_by_prefix, recommend
import streamlit as st

# BERT 모델과 토크나이저를 로드하는 함수
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = initialize_model(device)
    model.load_state_dict(torch.load('kobert_model.pth', map_location=device))
    model.eval()
    return model, tokenizer, device

# 문장을 입력받아 감정을 예측하고 영화를 추천하는 함수
def predict_and_recommend(sentence, model, tokenizer, device, movie_data, movie_hashmap):
    max_len = 64
    batch_size = 64
    data = [sentence, '0']
    dataset_another = [data]

    # BERTDataset을 사용하여 데이터셋을 만들고 DataLoader를 이용해 배치 처리
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len, True, False)
    test_dataloader = DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i.detach().cpu().numpy()
            emotion = np.argmax(logits)

            # 감정 예측 결과에 따라 다른 캐릭터 이미지, 대사, 감정 문자열 선택
            if emotion == 0:
                emotion_str = "기쁨이"
                color = "#FFD700"  # 금색
                char_img = "https://i.namu.wiki/i/bAzTR3Mc5tch1QOwro8aru4Z0AEtU4cmYvW6zoFiWJGzLBgd2R58SnlGP0HlRm1N3ig66piPRq6Zy_BNUGUtqA.webp"
                char_dialogue = "이 영화를 보면서 정말 행복한 순간을 만끽해봐요! 저와 함께 기쁨을 느껴봐요!"
                emotion_key = "Joy"
            elif emotion == 1:
                emotion_str = "슬픔이"
                color = "#1E90FF"  # 진회색
                char_img = "https://i.namu.wiki/i/VUwiIg58af_1wKoYrVTYdKXCMkQfNO8PbB32Rjf27M1dIPbwCEDwmCpTZqL8Pi2mbWKLqnFDUdkJTH_VN2xdbA.webp"
                char_dialogue = "이 영화를 보며 내 어깨에 기대어 울어도 괜찮아요. 저도 같이 있어줄게요."
                emotion_key = "Sadness"
            elif emotion == 2:
                emotion_str = "버럭이"
                color = "#FF4500"  # 주홍색
                char_img = "https://i.namu.wiki/i/yMwKPmyk_e9Ohk-IMpWiGdRxusUB_s18WQMCWoO22z4ot17wlHJpO1C2ZNWPfWW-t9J2hYdkHY0YBztI4JzoGg.webp"
                char_dialogue = "이 영화를 보면서 스트레스를 확 날려버려요! 같이 화내면서 속 시원해질 거예요!"
                emotion_key = "Anger"
            elif emotion == 3:
                emotion_str = "까칠이"
                color = "#32CD32"  # 라임색
                char_img = "https://i.namu.wiki/i/ZeGGZBUYHmppb5RTym8vJmnQulv-UhJygvrUMH_qXurQ08oZzRshu2sToDMet1NFeq4iwnkqY69Y_pKH4C2lmg.webp"
                char_dialogue = "이 영화를 보면서 짜증을 털어내봐요. 우리 함께 기분을 전환해요!"
                emotion_key = "Disgust"
            elif emotion == 4:
                emotion_str = "소심이"
                color = "#8A2BE2"  # 보라색
                char_img = "https://i.namu.wiki/i/IMJTGYaO0v6OYa5iro8bLfHuxdiXflvRQ4BdsMVOEvhFP2VBF74QAdMc4PFs-dJcYu-b9aRFeEnajUO1nDQeDg.webp"
                char_dialogue = "이 영화를 보면서 작은 용기를 키워봐요. 제가 응원할게요."
                emotion_key = "Fear"

            # 해당 감정에 맞는 영화 추천
            recommended_movies = recommend(emotion_key, movie_data, [], movie_hashmap)   

            return emotion_str, color, char_img, char_dialogue, recommended_movies

def main():
    # Streamlit 배경 색상 설정 (다크 모드)
    st.markdown("""
    <style>
    .stApp {
        background-color: #0F1116;
        color: #ffffff;
    }
    .title {
        font-size: 2.3em;
        color: #ffffff;
        text-align: center;
    }
    .subtitle {
        font-size: 0.9em;
        color: #ffffff;
        text-align: right;
    }
    .custom-text {
        line-height: 1.7;
        color: #ffffff;
        font-size: 1.0em;
    }
    .section {
        padding: 10px;
        margin: 10px 0;
    }
    .button {
        background-color: #FFD700;
        color: #0F1116;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 타이틀 및 정보 표시
    st.markdown("<h1 class='title'>🎬 인사이드 필름: 감정 기반 영화 추천 서비스 </h1>", unsafe_allow_html=True)
    # st.markdown("<p class='subtitle'>의용생체공학과 202138141 정수미</p>", unsafe_allow_html=True)

    # 프로젝트 설명
    st.markdown("""
    <div class='section'>
        <h5 class='custom-text'>
            영화 <인사이드 아웃>의 캐릭터들을 바탕으로 당신의 감정을 분석하고, 그에 맞는 영화를 추천해드립니다.<br>
            지금 기분을 이야기해보세요, 당신에게 딱 맞는 영화를 찾아드릴게요!
        </h5>
    </div>
    """, unsafe_allow_html=True)

    # 인사이드 아웃 GIF 이미지
    st.image("https://media1.tenor.com/m/UuxfrhK2g70AAAAC/yay-inside-out.gif", use_column_width=True)

    # 인사이드 아웃 명대사
    st.markdown("<blockquote class='custom-text' style='text-align:left;'><i>Emotions are something you can't give up</i></blockquote>", unsafe_allow_html=True)

    # 사용자 입력을 위한 텍스트 상자 초기화 및 버튼 클릭 상태 저장
    if 'input' not in st.session_state:
        st.session_state['input'] = ''
    if 'search_query' not in st.session_state:
        st.session_state['search_query'] = ''
    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = []

    # BERT 모델과 영화 데이터셋 로드
    model, tokenizer, device = load_model()
    movie_data, movie_hashmap, movie_trie = load_movie_data('./dataset/movies.csv')

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    
    # 사용자 입력을 받고 감정 분석 및 추천 버튼
    user_input = st.text_input("어떤 기분이신가요? 하고 싶은 말을 입력해주세요:", key="input", help=None, placeholder=None, disabled=False, label_visibility="visible")

    if st.button("분석 및 추천"):
        if user_input:
            # 검색 결과 초기화
            st.session_state['show_results'] = False
            st.session_state['search_results'] = []

            emotion, color, char_img, char_dialogue, recommended_movies = predict_and_recommend(user_input, model, tokenizer, device, movie_data, movie_hashmap)

            st.markdown(
                f"""
                <style>
                .speech-bubble {{
                    position: relative;
                    background: {color};
                    border-radius: .4em;
                    padding: 10px;
                    color: white;
                    margin: 20px 0;
                    display: inline-block;
                }}
                .speech-bubble::after {{
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 0;
                    width: 0;
                    height: 0;
                    border: 10px solid transparent;
                    border-right-color: {color};
                    border-left: 0;
                    border-top: 0;
                    margin-top: -5px;
                    margin-left: -10px;
                }}
                .recommend-movie {{
                    color: #fff;
                    padding: 5px 0;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"<h3 style='color:#fff;'>당신의 감정은 <span style='color:{color}; font-size: 1.3em;'>'{emotion}'</span>와 유사하게 느껴집니다.</h3>", unsafe_allow_html=True)
            st.image(char_img, width=300)  
            st.markdown(f"<div class='speech-bubble'>{char_dialogue}</div>", unsafe_allow_html=True)

            if recommended_movies:
                st.markdown("<h3 style='color: #ffffff;'>추천 영화</h3>", unsafe_allow_html=True)
                for title, release_time in recommended_movies:
                    st.markdown(f"<p style='color: #ffffff;'><b>{title} ({release_time})</b></p>", unsafe_allow_html=True)

                    # 영화에 대한 상세 정보는 확장 가능한 섹션에 숨김
                    if title in movie_hashmap:
                        movie_info = movie_hashmap[title]
                        with st.expander(f"영화 상세 정보"):
                            st.markdown(f"<p style='color: #ffffff;'>장르: {movie_info['genre']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>감독: {movie_info['director']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>출시 날짜: {movie_info['release_time']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>배급사: {movie_info['distributor']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>상영 시간: {movie_info['time']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>상영 등급: {movie_info['screening_rat']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>이전 감독 작품 관객수: {movie_info['dir_prev_bfnum']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>감독의 이전 작품 수: {movie_info['dir_prev_num']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>스텝 수: {movie_info['num_staff']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>주연 배우 수: {movie_info['num_actor']}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #ffffff;'>누적 관객 수: {movie_info['box_off_num']}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #ffffff;'>죄송합니다. 해당 감정에 맞는 추천 영화가 없습니다.</p>", unsafe_allow_html=True)
    
    if st.button("다시하기"):
        st.session_state.clear()
        st.experimental_rerun()

    st.markdown("</div><br>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    
    # 영화 검색 기능 추가
    search_query = st.text_input("원하는 영화를 검색해보세요:", key="search_query", help=None, placeholder=None, disabled=False, label_visibility="visible")
    if st.button("영화 검색"):
        if search_query:
            search_results = search_movies_by_prefix(movie_trie, search_query)
            st.session_state['search_results'] = search_results
            st.session_state['show_results'] = True
        else:
            st.session_state['show_results'] = False

    if st.session_state.get('show_results', False):
        if st.session_state.get('search_results', []):
            st.markdown("<h3 style='color: #ffffff;'>검색 결과</h3>", unsafe_allow_html=True)
            for movie_title in st.session_state['search_results']:
                st.markdown(f"<p style='color: #ffffff;'>{movie_title}</p>", unsafe_allow_html=True)
                # 각 영화에 대한 상세 정보를 보여주는 기능 추가 가능
        else:
            st.markdown("<p style='color: #ffffff;'>검색 결과가 없습니다.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
