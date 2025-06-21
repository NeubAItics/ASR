# Overview of the all the py scripts  
**whisp_t5.py** - whisper and t5-base  
works perfectly fine  
---------------------------------------------------------------------------
**bart_model.py** - whisper and bart/large  
not so good in terms of sumarization. *summary is too long*  
---------------------------------------------------------------------------
**w2v_bart.py** - wave2vec2 and bart/large  
very very poor transcription.  
- not sure is frequency was set correctly to match training frequency.  
- gpt said it is not very good at converational audio clips  
- may work better if we do preprocessing on audio clips to reduce noise.  
--------------------------------------------------------------------------------------


# Streamlit links  
`whisp_t5.py` - https://3pq9nvfhnfps6kvx75cqzc.streamlit.app/ *recommended*  
`bart_model.py` - https://nefc3ojbpt9bwrso5e2czg.streamlit.app/
