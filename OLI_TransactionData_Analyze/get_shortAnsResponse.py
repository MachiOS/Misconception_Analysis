import pandas as pd

# file_list = [
# '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Fall 2019/ds4736_tx_2024_1020_234729/ds4736_tx_All_Data_6897_2021_0609_103857.csv',
# '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Fall 2020/ds4734_tx_2024_1020_234528/ds4734_tx_All_Data_6895_2021_0609_094208.csv',
# '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Spring 2019/ds4732_tx_2024_1020_234221/ds4732_tx_All_Data_6893_2021_0608_174948.csv',
# '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Spring2020/ds4733_tx_2024_1020_234340/ds4733_tx_All_Data_6894_2021_0609_091342.csv',
# '/Users/machi/Desktop/OLI Biology Data _ALL/Betshune-Cookman INtro Biology/Spring2021/ds4735_tx_2024_1020_234632/ds4735_tx_All_Data_6896_2021_0609_101008.csv'
# ]

# dataset_list = [
#     'ds4736_tx_2024_1020_234729',
#     'ds4734_tx_2024_1020_234528',
#     'ds4732_tx_2024_1020_234221',
#     'ds4733_tx_2024_1020_234340',
#     'ds4735_tx_2024_1020_234632'
# ]

# file_list = [
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.1 Spring 2019/ds4764_tx_2024_1023_184235/ds4764_tx_All_Data_6934_2021_0622_125947.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.4 Summer 2019/ds4765_tx_2024_1021_004920/ds4765_tx_All_Data_6935_2021_0623_161918.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.5 Fall 2019/ds4767_tx_2024_1021_005206/ds4767_tx_All_Data_6937_2021_0623_173940.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.6 Spring 2020/ds4766_tx_2024_1021_005044/ds4766_tx_All_Data_6936_2021_0623_165916.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.7 Fall 2020/ds4784_tx_2024_1021_005423/ds4784_tx_All_Data_6948_2021_0625_165224.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.7 Summer 2020/ds4817_tx_2024_1021_005635/ds4817_tx_All_Data_6951_2021_0627_164706.csv',
#     '/Users/machi/Desktop/OLI Biology Data _ALL/Heartland Community Colleage Engaging Biology/1.9 Spring 2021/ds4825_tx_2024_1021_005809/ds4825_tx_All_Data_6953_2021_0923_001631.csv'
# ]

# dataset_list = [
#     'ds4764_tx_2024_1023_184235',
#     'ds4765_tx_2024_1021_004920',
#     'ds4767_tx_2024_1021_005206',
#     'ds4766_tx_2024_1021_005044',
#     'ds4784_tx_2024_1021_005423',
#     'ds4817_tx_2024_1021_005635',
#     'ds4825_tx_2024_1021_005809'
    
# ]

file_list = [
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/ALMAP fall 2013/ds959_tx_2024_1021_000225/ds959_tx_All_Data_2417_2019_0626_211138.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/ALMAP Spring 2014/ds960_tx_2024_1021_001400/ds960_tx_All_Data_2418_2019_0626_221116.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/ALMAP spring 2014 DS 960/ds1934_tx_2024_1021_003344/ds1934_tx_All_Data_3679_2021_0820_054350.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/Fall 2012 (complete)/ds618_tx_2024_1020_235934/ds618_tx_All_Data_1902_2017_0704_062505.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/Introduction to Biology 1.0 Spring 2018 part 1 of 6/ds2647_tx_2024_1023_185742/ds2647_tx_All_Data_4537_2018_0904_232146.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/Introduction to Biology 1.0 Summer 2018 part 1 of 3/ds2639_tx_2024_1021_003549/ds2639_tx_All_Data_4530_2018_0831_154504.csv',
    '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/Introduction to Biology Fall 2014 (DNA-2)/ds1187_tx_2024_1021_001617/ds1187_tx_All_Data_2761_2017_0224_052343.csv'
]

dataset_list = [
    'ds959_tx_2024_1021_000225',
    'ds960_tx_2024_1021_001400',
    'ds1934_tx_2024_1021_003344',
    'ds618_tx_2024_1020_235934',
    'ds2647_tx_2024_1023_185742',
    'ds2639_tx_2024_1021_003549',
    'ds1187_tx_2024_1021_001617'
]





output_df = ''
df_list = []
for i,file in enumerate(file_list):
    df = pd.read_csv(file)
    
    sub_df = df[df['Action']=='UpdateShortAnswer']
    sub_df['dataset'] = dataset_list[i]
    df_list.append(sub_df)

out_df  = pd.concat(df_list)

out_df.to_csv( '/Users/machi/Desktop/OLI Biology Data _ALL/OLI Biology/all_shortAns_concat/OLI_ShortAns.csv',index=False)
    
    
        
        
        
        
    
    