from tick_model import Tick_model

v_width = 0.5
v_prominence = v_width*2
sber = Tick_model('1min.txt', v_width=v_width, v_prominence=v_prominence)
sber.load_vectors("all-vectors-100k.csv")
lag_width = 10
sber.create_model_for_vectors(30)