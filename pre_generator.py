import pandas as pd
from AI.generator import generate

GENERATE_NUM = 100000
initials = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふめほまみむめもらりるれろわ'

def main():
    ku_list = []
    # あいうえお順でGENERATE_NUMで指定された数だけ句を生成する
    for idx in range(GENERATE_NUM):
        initial = initials[idx % len(initials)]
        AI_ku = generate(initial)
        ku_list.append(AI_ku)
    
    # 生成した句の保存
    df = pd.DataFrame({'ku': ku_list})
    df.to_csv('./data/ku_list.csv')
        
if __name__ == '__main__':
    main()
