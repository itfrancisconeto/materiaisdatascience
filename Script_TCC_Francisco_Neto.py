# Importa as bibliotecas necessarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#---#
################################# FUNÇÕES E VARIÁVEIS DE APOIO #################################

#Variavel global
anocorrente = 2020

#Funcao para substituicao de caracteres
def replacecaracter(df, column, carac_1, carac_2):
    df[column] = df[column].str.replace(carac_1, carac_2)
    return df

#Funcao para tratamento de valores NaN
def replacenan(df, column):
    df[column] = df[column].fillna(-1)
    return df

#Funcao para alteracao de tipo de dado para ponto flutuante
def replacetypetofloat(df, column):
    df[column] = df[column].astype(float)
    #print(df[column].dtypes)
    return df

#Funcao para exclusão de linha mediante condição da duração dos rounds ser maior ou igual a 3
def delrowinvalidround(df):
    column = 'Format'    
    for key, value in df[column].iteritems():
        item = value.split(' ')[0]        
        df[column].at[key] = str(item)   
    df = df.drop(df[(df[column] != '3') & (df[column] != '5')].index)
    df.columns = df.columns.str.replace('Format','Rounds')    
    return df

#Funcao para alteracao de tipo de dado para inteiro
def replacetypetoint(df, column):
    df[column] = df[column].astype(int)
    #print(df[column].dtypes)
    return df

#Funcao para alteracao de tipo de dado para string
def replacetypetostr(df, column):
    df[column] = df[column].astype(str)
    #print(df[column].dtypes)
    return df

#Funcao para calculo da idade do lutador
def agefither(df, column):
    df[column] = df[column].str.slice(start=-4)
    df[column] = df[column].fillna(-1)
    df[column] = df[column].astype(int)
    for key, value in df[column].iteritems(): 
      if value != -1:
        value = anocorrente - value
        df[column].at[key] = value      
    return df

#Funcao para plotagem de grafico da frequencia das posturas dos lutadores
def frequencystance(df,stc_open,stc_orthodox,stc_sideways,stc_southpaw,stc_switch):
    grupos = [stc_open,stc_orthodox,stc_sideways,stc_southpaw,stc_switch]
    vlr_stc_open = len(df.loc[df[stc_open] == 1])
    vlr_stc_orthodox =  len(df.loc[df[stc_orthodox] == 1])
    vlr_stc_sideways =  len(df.loc[df[stc_sideways] == 1])
    vlr_stc_southpaw =  len(df.loc[df[stc_southpaw] == 1])
    vlr_stc_switch =  len(df.loc[df[stc_switch] == 1])
    valores = [vlr_stc_open,vlr_stc_orthodox,vlr_stc_sideways,vlr_stc_southpaw,vlr_stc_switch]
    plt.figure(figsize=(12,10))
    plt.title('Frequencia das posturas')
    plt.xlabel('Posturas')
    plt.ylabel('Frequencias')
    plt.bar(grupos,valores)
    print('Valores das frequencias: ', valores)
    plt.show()  

#Funcao para plotagem de grafico da correlação entre os atributos dos lutadores
def figthercorrelation(df,height,weight,reach,age,r_stc_open,r_stc_orthodox,r_stc_sideways,r_stc_southpaw,r_stc_switch,b_stc_open,b_stc_orthodox,b_stc_sideways,b_stc_southpaw,b_stc_switch):
    df.drop(height, axis=1, inplace=True)
    df.drop(weight, axis=1, inplace=True)
    df.drop(reach, axis=1, inplace=True)
    df.drop(age, axis=1, inplace=True)
    df.drop(r_stc_open, axis=1, inplace=True)
    df.drop(r_stc_orthodox, axis=1, inplace=True)
    df.drop(r_stc_sideways, axis=1, inplace=True)
    df.drop(r_stc_southpaw, axis=1, inplace=True)
    df.drop(r_stc_switch, axis=1, inplace=True)
    df.drop(b_stc_open, axis=1, inplace=True)
    df.drop(b_stc_orthodox, axis=1, inplace=True)
    df.drop(b_stc_sideways, axis=1, inplace=True)
    df.drop(b_stc_southpaw, axis=1, inplace=True)
    df.drop(b_stc_switch, axis=1, inplace=True)
    plt.figure(figsize=(12,10))
    df_corr_ = df.corr()
    ax_ = sns.heatmap(df_corr_, annot=True)
    bottom, top = ax_.get_ylim()
    ax_.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

#Funcao para plotagem de grafico da frequencia das idades do lutador pela categoria de peso
def frequencyagebycategory(df,column_age,category):
    idx = df[df[column_age] == -1].index
    df.drop(idx, inplace=True)    
    df = df.drop(df[(df['Fight_type'] != category)].index)
    df = replacetypetoint(df, column_age)
    sns.distplot(df[column_age], kde=False, color='blue', bins=25)
    plt.title('Frequencia das idades')
    plt.xlabel('Idades')
    plt.ylabel('Frequencias')
    plt.show()

#Funcao para plotagem de grafico da frequencia das idades do lutador pela categoria de peso
def frequencyreachbycategory(df,column_reach,category):
    idx = df[df[column_reach] == -1].index
    df.drop(idx, inplace=True)    
    df = df.drop(df[(df['Fight_type'] != category)].index)
    df = replacetypetoint(df, column_reach)
    sns.distplot(df[column_reach], kde=False, color='blue', bins=12)
    plt.title('Frequencia das envergaduras')
    plt.xlabel('Envergaduras')
    plt.ylabel('Frequencias')
    plt.show()

#Funcao para plotagem de grafico boxplot dos atributos dos lutadures
def boxplotattributes(df,column):
    plt.figure(figsize=(12,10))
    sns.boxplot( y=df[column],width=0.5);
    plt.show()

#Funcao para calculo da idade do lutador cuja DOB não foi informado
def agefithermean(df,column_age):    
    idx = df[df[column_age] == -1].index
    df.drop(idx, inplace=True)
    df = df.groupby(by=['Fight_type'])[column_age].agg(pd.Series.mean).to_frame()
    idx_to_column = df.index.values
    df.insert(0,column='Fight_type',value = idx_to_column)
    df.reset_index(drop=True, inplace=True)
    return df

#Funcao para atribuir a media da idade por categoria para os valores de idade missing
def agefithermissing(df_orig,df_dest,column_orig,column_dest):
    for key_dest, value_dest in df_dest['Fight_type'].iteritems():
        for key_orig, value_orig in df_orig['Fight_type'].iteritems():
            if value_dest == value_orig:
               idx = df_orig[df_orig['Fight_type']==value_orig].index.values.item()
               item_orig = df_orig.loc[idx,[column_orig]]              
               item_dest = df_dest.loc[key_dest,column_dest]
               if item_dest == -1:
                   df_dest[column_dest].at[key_dest] = item_orig
               break    
    return df_dest 

#Funcao para calculo da enveregadura do lutador cuja Reach não foi informado
def reachfithermean(df,column_reach):    
    idx = df[df[column_reach] == -1].index
    df.drop(idx, inplace=True)
    df = df.groupby(by=['Fight_type'])[column_reach].agg(pd.Series.mean).to_frame()
    idx_to_column = df.index.values
    df.insert(0,column='Fight_type',value = idx_to_column)
    df.reset_index(drop=True, inplace=True)
    return df

#Funcao para atribuir a media da idade por categoria para os valores de idade missing
def reachfithermissing(df_orig,df_dest,column_orig,column_dest):
    for key_dest, value_dest in df_dest['Fight_type'].iteritems():
        for key_orig, value_orig in df_orig['Fight_type'].iteritems():
            if value_dest == value_orig:
               idx = df_orig[df_orig['Fight_type']==value_orig].index.values.item()
               item_orig = df_orig.loc[idx,[column_orig]]
               item_dest = df_dest.loc[key_dest,column_dest]
               if item_dest == -1:
                   df_dest[column_dest].at[key_dest] = item_orig
               break
    return df_dest

#Funcao para popular atributos do merge dos dataframe
def populamergedataframe(df_orig,df_dest,column_orig,column_dest,fighter_dest):    
    for key_dest, value_dest in df_dest[fighter_dest].iteritems():
        for key_orig, value_orig in df_orig['fighter_name'].iteritems():
            if value_dest == value_orig:
               idx = df_orig[df_orig['fighter_name']==value_orig].index.values.item()
               item_orig = df_orig.loc[idx,[column_orig]]              
               df_dest[column_dest].at[key_dest] = item_orig
               break    
    return df_dest

#Funcao para merge dos dataframe
def mergedataframe(df_orig, df_dest):
    df_dest['R_Height'] = 0.0
    df_dest['R_Weight'] = 0
    df_dest['R_Reach'] = 0
    df_dest['R_Age'] = 0
    df_dest['R_Stance_Open'] = 0
    df_dest['R_Stance_Orthodox'] = 0
    df_dest['R_Stance_Sideways'] = 0
    df_dest['R_Stance_Southpaw'] = 0
    df_dest['R_Stance_Switch'] = 0
    df_dest['B_Height'] = 0.0
    df_dest['B_Weight'] = 0
    df_dest['B_Reach'] = 0
    df_dest['B_Age'] = 0
    df_dest['B_Stance_Open'] = 0
    df_dest['B_Stance_Orthodox'] = 0
    df_dest['B_Stance_Sideways'] = 0
    df_dest['B_Stance_Southpaw'] = 0
    df_dest['B_Stance_Switch'] = 0    
    # Popula atributo R_Height
    print('\n Processando 1/18 >>> Populando atributo R_Height no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Height', 'R_Height', 'R_fighter')      
    # Popula atributo R_Weight
    print('\n Processando 2/18 >>> Populando atributo R_Weight no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Weight', 'R_Weight', 'R_fighter')
    # Popula atributo R_Reach
    print('\n Processando 3/18 >>> Populando atributo R_Reach no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Reach', 'R_Reach', 'R_fighter')           
    # Popula atributo R_Age
    print('\n Processando 4/18 >>> Populando atributo R_Age no dataframe de destino')
    df_dest = populamergedataframe(df_orig,df_dest,'DOB','R_Age','R_fighter')
    # Popula atributo R_Stance_Open
    print('\n Processando 5/18 >>> Populando atributo R_Stance_Open no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Open Stance', 'R_Stance_Open', 'R_fighter')
    # Popula atributo R_Stance_Orthodox
    print('\n Processando 6/18 >>> Populando atributo R_Stance_Orthodox no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Orthodox', 'R_Stance_Orthodox', 'R_fighter')
    # Popula atributo R_Stance_Sideway
    print('\n Processando 7/18 >>> Populando atributo R_Stance_Sideways no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Sideways', 'R_Stance_Sideways', 'R_fighter')
    # Popula atributo R_Stance_Southpaw
    print('\n Processando 8/18 >>> Populando atributo R_Stance_Southpaw no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Southpaw', 'R_Stance_Southpaw', 'R_fighter')
    # Popula atributo R_Stance_Switch
    print('\n Processando 9/18 >>> Populando atributo R_Stance_Switch no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Switch', 'R_Stance_Switch', 'R_fighter')
    # Popula atributo B_Height
    print('\n Processando 10/18 >>> Populando atributo B_Height no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Height', 'B_Height', 'B_fighter')
    # Popula atributo B_Weight
    print('\n Processando 11/18 >>> Populando atributo B_Weight no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Weight', 'B_Weight', 'B_fighter')
    # Popula atributo B_Reach
    print('\n Processando 12/18 >>> Populando atributo B_Reach no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Reach', 'B_Reach', 'B_fighter')           
    # Popula atributo B_Age
    print('\n Processando 13/18 >>> Populando atributo B_Age no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'DOB', 'B_Age', 'B_fighter')
    # Popula atributo B_Stance_Open
    print('\n Processando 14/18 >>> Populando atributo B_Stance_Open no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Open Stance', 'B_Stance_Open', 'B_fighter')
    # Popula atributo B_Stance_Orthodox
    print('\n Processando 15/18 >>> Populando atributo B_Stance_Orthodox no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Orthodox', 'B_Stance_Orthodox', 'B_fighter')
    # Popula atributo B_Stance_Sideway
    print('\n Processando 16/18 >>> Populando atributo B_Stance_Sideways no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Sideways', 'B_Stance_Sideways', 'B_fighter')
    # Popula atributo B_Stance_Southpaw
    print('\n Processando 17/18 >>> Populando atributo B_Stance_Southpaw no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Southpaw', 'B_Stance_Southpaw', 'B_fighter')
    # Popula atributo B_Stance_Switch
    print('\n Processando 18/18 >>> Populando atributo B_Stance_Switch no dataframe de destino')
    df_dest = populamergedataframe(df_orig, df_dest, 'Stance_Switch', 'B_Stance_Switch', 'B_fighter')
    # Retorno da função
    return df_dest

#Funcao para preparar o dataframe para ML
def preparedftoml(df):
    df.drop('R_fighter', axis=1, inplace=True)
    df.drop('B_fighter', axis=1, inplace=True)
    df.drop('Fight_type', axis=1, inplace=True)
    df['Target'] = 0 
    df = replacetypetoint(df,'Rounds')
    #df.info()    
    df['Target'] = df['Rounds'] - df['last_round']    
    for key, value in df['Target'].iteritems(): 
      if value == 0:
        df['Target'].at[key] = 1
      else:
        df['Target'].at[key] = 0          
    return df

#Funcao para exibir a quantidade de lutadore por categoria de peso
def countfigherbycategory(df):
    df = df.filter(['Fight_type','R_fighter'],axis=1)
    df.columns = df.columns.str.replace('Fight_type','Category')
    df.columns = df.columns.str.replace('R_fighter','QtdFither')    
    for key, value in df['QtdFither'].iteritems():        
        df['QtdFither'].at[key] = 1      
    df = df.groupby(by=['Category'])['QtdFither'].agg(sum).to_frame()
    df = df.reset_index()
    print(df.sort_values(by='QtdFither', ascending=False))
    
################################# INICIO DO PROCESSO #################################   
    
##### IMPORTAÇÃO DO DATAFRAME COM DETALHES DOS LUTADORES (raw_fighter_details)
print('\n INICIO DO PROCESSO')
print('\n 1) Importa para dataframe o arquivo com detalhes dos lutadores')
# Faz a leitura do arquivo CSV
df_1_fighter = pd.read_csv("Dataset/raw_fighter_details.csv",encoding="ISO-8859-1")
# Faz uma copia do dataframe original
df_2_fighter_proc = df_1_fighter.copy()

# Verifica a existencia de dados faltantes
print('\n 2) Verifica a existencia de dados faltantes em df_2_fighter_proc:')
print(df_2_fighter_proc.isnull().sum())
print()

##### SUBSTITUICAO DE CARACTERES E TIPOS DE DADOS
print('\n 3) Substitui caracteres e tipos de dados no dataframe')
# Substitui caracteres da coluna Height (Altura)
df_2_fighter_proc = replacecaracter(df_2_fighter_proc,'Height','\' ','.')
df_2_fighter_proc = replacecaracter(df_2_fighter_proc,'Height','\"','')
# Substitui caracteres da coluna Weight (Peso)
df_2_fighter_proc = replacecaracter(df_2_fighter_proc,'Weight',' lbs.','')
# Substitui caracteres da coluna Reach (Envergadura)
df_2_fighter_proc = replacecaracter(df_2_fighter_proc,'Reach','\"','')
# Substitui os valores NaN da coluna Height  (Altura) por -1 (menos um)
df_2_fighter_proc = replacenan(df_2_fighter_proc,'Height')
# Substitui os valores NaN da coluna Weight  (Peso) por -1 (menos um)
df_2_fighter_proc = replacenan(df_2_fighter_proc,'Weight')
# Substitui os valores NaN da coluna Reach (Envergadura) por -1 (menos um)
df_2_fighter_proc = replacenan(df_2_fighter_proc,'Reach')
# Converte a coluna Height (Altura) de string para float
df_2_fighter_proc = replacetypetofloat(df_2_fighter_proc,'Height')
# Converte a coluna Weight (Peso) de string para inteiro
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Weight')
# Converte a coluna Reach (Envergadura) de string para inteiro
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Reach')
# Converte a coluna Stance (Postura) de object para string
df_2_fighter_proc = replacetypetostr(df_2_fighter_proc,'Stance')

##### GERA COLUNAS DUMMIES PARA CATEGORIAS DE POSTURA DO LUTADOR
print('\n 4) Gera no dataframe colunas dummines para categoria de postura do lutador')
# Substitui os valores da coluna Stance (Postura) criando uma nova coluna para cada valor em Stance
df_2_fighter_proc = pd.concat([df_2_fighter_proc.drop('Stance', axis=1), pd.get_dummies(df_2_fighter_proc['Stance'],prefix='Stance')], axis=1)
## Converte as colunas Stance para inteiro
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Stance_Open Stance')
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Stance_Orthodox')
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Stance_Sideways')
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Stance_Southpaw')
df_2_fighter_proc = replacetypetoint(df_2_fighter_proc,'Stance_Switch')

##### CALCULA A IDADE DO LUTADOR
print('\n 5) Calcula a idade lutador')
#Calcula a idade do lutador
df_2_fighter_proc = agefither(df_2_fighter_proc, 'DOB')

##### IMPORTAÇÃO DO DATAFRAME COM DETALHES DAS LUTAS (raw_total_fight_data)
print('\n 6) Importa para dataframe o arquivo com detalhes das lutas')
#Faz a leitura do arquivo CSV
df_3_total_fight = pd.read_csv("Dataset/raw_total_fight_data.csv",sep=';',encoding="ISO-8859-1")
# Exibe as informacoes do dataframe
#df_3_total_fight.info()
#Faz uma copia do dataframe original
df_4_total_fight_proc = df_3_total_fight.copy()
#Verifica a existencia de dados faltantes
#print('\n>>> Verifica a existencia de dados faltantes em df_4_total_fight_proc_proc:')
#print(df_4_total_fight_proc.isnull().sum())

#Exibe a quantidade de lutadores por categoria em ordem decrescente
print('\n 7) Exibe a quantidade de lutadores por categoria em ordem decrescente:\n')
countfigherbycategory(df_4_total_fight_proc)

##### EXCLUSÃO DE COLUNAS DESNECESSÁRIAS
print('\n 8) Exclui atributos desnecessários ao processo')
#Exclui colunas desnecessárias para a análise
df_4_total_fight_proc.drop('R_KD', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_KD', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_SIG_STR.', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_SIG_STR.', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_SIG_STR_pct', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_SIG_STR_pct', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_TOTAL_STR.', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_TOTAL_STR.', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_TD', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_TD', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_TD_pct', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_TD_pct', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_SUB_ATT', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_SUB_ATT', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_PASS', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_PASS', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_REV', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_REV', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_HEAD', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_HEAD', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_BODY', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_BODY', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_LEG', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_LEG', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_DISTANCE', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_DISTANCE', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_CLINCH', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_CLINCH', axis=1, inplace=True)
df_4_total_fight_proc.drop('R_GROUND', axis=1, inplace=True)
df_4_total_fight_proc.drop('B_GROUND', axis=1, inplace=True)
df_4_total_fight_proc.drop('win_by', axis=1, inplace=True)
df_4_total_fight_proc.drop('last_round_time', axis=1, inplace=True)
df_4_total_fight_proc.drop('Referee', axis=1, inplace=True)
df_4_total_fight_proc.drop('date', axis=1, inplace=True)
df_4_total_fight_proc.drop('location', axis=1, inplace=True)
df_4_total_fight_proc.drop('Winner', axis=1, inplace=True)
#Exclui linhas cujo campo de duração total de rounds (Format) tenha valor menor do que 3
df_4_total_fight_proc = delrowinvalidround(df_4_total_fight_proc)

##### MERGE DOS DATAFRAMES
print('\n 9) Unifica os dataframes de detalhes dos lutadores e das lutas')
# Unifica os dois dataframes
df_4_total_fight_proc = mergedataframe(df_2_fighter_proc,df_4_total_fight_proc)
# Verifica a existencia de dados faltantes
#print('\n>>> Verifica a existencia de dados faltantes em df_4_total_fight_proc:')
#print(df_4_total_fight_proc.isnull().sum())
print('\n 10) Visualiza a distribuição de dados do atributo R_Height:\n')
boxplotattributes(df_4_total_fight_proc,'R_Height')
print('\n 11) Visualiza a distribuição de dados do atributo R_Weight:\n')
boxplotattributes(df_4_total_fight_proc,'R_Weight')
print('\n 12) Visualiza a distribuição de dados do atributo R_Reach:\n')
boxplotattributes(df_4_total_fight_proc,'R_Reach')
print('\n 13) Visualiza a distribuição de dados do atributo R_Age:\n')
boxplotattributes(df_4_total_fight_proc,'R_Age')
print('\n 14) Visualiza a distribuição de dados do atributo B_Height:\n')
boxplotattributes(df_4_total_fight_proc,'B_Height')
print('\n 15) Visualiza a distribuição de dados do atributo B_Weight:\n')
boxplotattributes(df_4_total_fight_proc,'B_Weight')
print('\n 16) Visualiza a distribuição de dados do atributo B_Reach:\n')
boxplotattributes(df_4_total_fight_proc,'B_Reach')
print('\n 17) Visualiza a distribuição de dados do atributo B_Age:\n')
boxplotattributes(df_4_total_fight_proc,'B_Age')

##### CALCULO DAS IDADES MISSING DOS LUTADORES PELA MEDIA DE CADA CATEGORIA DE PESO
print('\n 18) Calcula as idades missing dos lutadores pela média da categoria de peso')
    
df_5a_R_Age_by_Category = df_4_total_fight_proc.filter(['Fight_type','R_Age'],axis=1)
df_5a_R_Age_by_Category = df_5a_R_Age_by_Category[(df_5a_R_Age_by_Category.R_Age >= 0)]   
df_5a_R_Age_by_Category = df_5a_R_Age_by_Category.drop(df_5a_R_Age_by_Category[(df_5a_R_Age_by_Category['Fight_type'] != 'Lightweight Bout')].index)

df_6a_B_Age_by_Category = df_4_total_fight_proc.filter(['Fight_type','B_Age'],axis=1)
df_6a_B_Age_by_Category = df_6a_B_Age_by_Category[(df_6a_B_Age_by_Category.B_Age >= 0)]  
df_6a_B_Age_by_Category = df_6a_B_Age_by_Category.drop(df_6a_B_Age_by_Category[(df_6a_B_Age_by_Category['Fight_type'] != 'Welterweight Bout')].index)

print (df_5a_R_Age_by_Category.describe())
print()
print (df_6a_B_Age_by_Category.describe())

#Extrai as colunas categoria e idade dos lutadores
df_5_R_Age_by_Category = df_4_total_fight_proc.filter(['Fight_type','R_Age'],axis=1)
#Visualiza a frequencia das idades dos lutadores R pela categoria de peso
print('\n 18.1) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Lightweight Bout:\n')
frequencyagebycategory(df_5_R_Age_by_Category,'R_Age','Lightweight Bout')
print('\n 18.2) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Welterweight Bout:\n')
frequencyagebycategory(df_5_R_Age_by_Category,'R_Age','Welterweight Bout')
print('\n 18.3) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Middleweight Bout:\n')
frequencyagebycategory(df_5_R_Age_by_Category,'R_Age','Middleweight Bout')
#Identifica a media das idades por categoria
df_5_R_Age_by_Category = agefithermean(df_5_R_Age_by_Category,'R_Age')
#Atribui a media da idade por categoria para os valores de idade missing
df_4_total_fight_proc = agefithermissing(df_5_R_Age_by_Category,df_4_total_fight_proc,'R_Age','R_Age')
#Extrai as colunas categoria e idade dos lutadores
df_6_B_Age_by_Category = df_4_total_fight_proc.filter(['Fight_type','B_Age'],axis=1)
#Visualiza a frequencia das idades dos lutadores B pela categoria de peso
print('\n 18.4) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Lightweight Bout:\n')
frequencyagebycategory(df_6_B_Age_by_Category,'B_Age','Lightweight Bout')
print('\n 18.5) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Welterweight Bout:\n')
frequencyagebycategory(df_6_B_Age_by_Category,'B_Age','Welterweight Bout')
print('\n 18.6) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Middleweight Bout:\n')
frequencyagebycategory(df_6_B_Age_by_Category,'B_Age','Middleweight Bout')
#Identifica a media das idades por categoria
df_6_B_Age_by_Category = agefithermean(df_6_B_Age_by_Category,'B_Age')
#Atribui a media da idade por categoria para os valores de idade missing
df_4_total_fight_proc = agefithermissing(df_6_B_Age_by_Category,df_4_total_fight_proc,'B_Age','B_Age')

##### CALCULO DAS ENVERGADURAS (REACH) MISSING DOS LUTADORES PELA MEDIA DE CADA CATEGORIA DE PESO
print('\n 19) Calcula as envergaduras missing dos lutadores pela media da categoria de peso')
#Extrai as colunas categoria e envergadura dos lutadores
df_7_R_Reach_by_Category = df_4_total_fight_proc.filter(['Fight_type','R_Reach'],axis=1)
#Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso
print('\n 19.1) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Lightweight Bout:\n')
frequencyreachbycategory(df_7_R_Reach_by_Category,'R_Reach','Lightweight Bout')
print('\n 19.2) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Welterweight Bout:\n')
frequencyreachbycategory(df_7_R_Reach_by_Category,'R_Reach','Welterweight Bout')
print('\n 19.3) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Middleweight Bout:\n')
frequencyreachbycategory(df_7_R_Reach_by_Category,'R_Reach','Middleweight Bout')
#Identifica a media das envergaduras por categoria
df_7_R_Reach_by_Category = reachfithermean(df_7_R_Reach_by_Category,'R_Reach')
#Atribui a media da envergadura por categoria para os valores de envergadura missing
df_4_total_fight_proc = reachfithermissing(df_7_R_Reach_by_Category,df_4_total_fight_proc,'R_Reach','R_Reach')
#Extrai as colunas categoria e envergadura dos lutadores
df_8_B_Reach_by_Category = df_4_total_fight_proc.filter(['Fight_type','B_Reach'],axis=1)
#Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso
print('\n 19.4) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Lightweight Bout:\n')
frequencyreachbycategory(df_8_B_Reach_by_Category,'B_Reach','Lightweight Bout')
print('\n 19.5) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Welterweight Bout:\n')
frequencyreachbycategory(df_8_B_Reach_by_Category,'B_Reach','Welterweight Bout')
print('\n 19.6) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Middleweight Bout:\n')
frequencyreachbycategory(df_8_B_Reach_by_Category,'B_Reach','Middleweight Bout')
#Identifica a media das envergaduras por categoria
df_8_B_Reach_by_Category = reachfithermean(df_8_B_Reach_by_Category,'B_Reach')
#Atribui a media da envergadura por categoria para os valores de envergadura missing
df_4_total_fight_proc = reachfithermissing(df_8_B_Reach_by_Category,df_4_total_fight_proc,'B_Reach','B_Reach')
#Exclui os valores de envergadura cuja media pela categoria nao foi identificada
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Reach != 0) & (df_4_total_fight_proc.R_Reach != -1)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.B_Reach != 0) & (df_4_total_fight_proc.B_Reach != -1)]
#Exclui os valores de idade cuja media pela categoria nao foi identificada
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Age != 0) & (df_4_total_fight_proc.R_Age != -1)]
#Exclui os valores de altura não foi identificada
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Height != 0) & (df_4_total_fight_proc.R_Height != -1)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.B_Height != 0) & (df_4_total_fight_proc.B_Height != -1)]
#Elimina os outliers do atributo R_Weight
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Weight <= 245)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Age >= 23) & (df_4_total_fight_proc.R_Age <= 50)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.R_Reach >= 63) & (df_4_total_fight_proc.R_Reach <= 82)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.B_Weight <= 245)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.B_Age >= 24) & (df_4_total_fight_proc.B_Age <= 48)]
df_4_total_fight_proc = df_4_total_fight_proc[(df_4_total_fight_proc.B_Reach >= 64) & (df_4_total_fight_proc.B_Reach <= 80)]

###### MACHINE LEARNING
#Inclui no dataframe o atributo target
print('\n 20) Incluiu no dataframe o atributo Target')
df_4_total_fight_proc = preparedftoml(df_4_total_fight_proc)
df_4_total_fight_proc.drop('last_round', axis=1, inplace=True)
df_4_total_fight_proc.drop('Rounds', axis=1, inplace=True)
#Visualiza a correlação em gráfico para cada lutador
print('\n 21) Exibe a correlação entre os atributos do lutador R:')
df_4_total_fight_proc_R = df_4_total_fight_proc.copy()
figthercorrelation (df_4_total_fight_proc_R,'B_Height','B_Weight','B_Reach','B_Age','B_Stance_Open','B_Stance_Orthodox','B_Stance_Sideways','B_Stance_Southpaw','B_Stance_Switch','R_Stance_Open','R_Stance_Orthodox','R_Stance_Sideways','R_Stance_Southpaw','R_Stance_Switch')
print('\n 22) Exibe a correlação entre os atributos do lutador B:')
df_4_total_fight_proc_B = df_4_total_fight_proc.copy()
figthercorrelation (df_4_total_fight_proc_B,'R_Height','R_Weight','R_Reach','R_Age','B_Stance_Open','B_Stance_Orthodox','B_Stance_Sideways','B_Stance_Southpaw','B_Stance_Switch','R_Stance_Open','R_Stance_Orthodox','R_Stance_Sideways','R_Stance_Southpaw','R_Stance_Switch')
#Verifica como os dados da varíavel predita estão distribuídos
print('\n 23) Exibe a  distribuição de  dados da variável predita:')
num_true = len(df_4_total_fight_proc.loc[df_4_total_fight_proc['Target'] == 1])
num_false = len(df_4_total_fight_proc.loc[df_4_total_fight_proc['Target'] == 0])
print("\nNúmero de Casos Verdadeiros: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Número de Casos Falsos     : {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))
#print('\n>>> Verifica a distribuição de dados dos atributos populados no merge:')
print('\n 24) Visualiza a distribuição de dados do atributo R_Height:\n')
boxplotattributes(df_4_total_fight_proc,'R_Height')
print('\n 25) Visualiza a distribuição de dados do atributo R_Weight:\n')
boxplotattributes(df_4_total_fight_proc,'R_Weight')
print('\n 26) Visualiza a distribuição de dados do atributo R_Reach:\n')
boxplotattributes(df_4_total_fight_proc,'R_Reach')
print('\n 27) Visualiza a distribuição de dados do atributo R_Age:\n')
boxplotattributes(df_4_total_fight_proc,'R_Age')
print('\n 28) Visualiza a distribuição de dados do atributo B_Height:\n')
boxplotattributes(df_4_total_fight_proc,'B_Height')
print('\n 29) Visualiza a distribuição de dados do atributo B_Weight:\n')
boxplotattributes(df_4_total_fight_proc,'B_Weight')
print('\n 30) Visualiza a distribuição de dados do atributo B_Reach:\n')
boxplotattributes(df_4_total_fight_proc,'B_Reach')
print('\n 31) Visualiza a distribuição de dados do atributo B_Age:\n')
boxplotattributes(df_4_total_fight_proc,'B_Age')
#Prepara dataframe para ML
print('\n 32) Prepara o dataframe unificado para Machine Learning')
df_9_total_fight_ml = df_4_total_fight_proc.copy()
#Cria modelo de ML
print('\n 32.1) Separa os dados entre treino e teste')
X = df_9_total_fight_ml.drop('Target', axis=1)
Y = df_9_total_fight_ml['Target']
#Definindo a taxa de split
split_test_size = 0.3
#Criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size)
print('\n 32.2) Aplica os algoritmos de Machine Learning:')

#Utilizando o classificador Naive Bayes
print('\nModelo 1 - Classificador Naive Bayes:\n')
#Criando o modelo
modelo_v1 = GaussianNB()
#Treinando o modelo
modelo_v1.fit(X_treino, Y_treino.ravel())
#Verificando a exatidão no modelo nos dados de treino

nb_predict_train = modelo_v1.predict(X_treino)
print("Exatidão Dados Treino (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, nb_predict_train)))
print()
#Criando uma Confusion Matrix dos dados de treino
matrix = metrics.confusion_matrix(Y_treino, nb_predict_train, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_treino,nb_predict_train,rownames=['Real'],colnames=['Predito'],margins=True))
print()
#Verificando a exatidão no modelo nos dados de teste
nb_predict_test = modelo_v1.predict(X_teste)
print("\nExatidão Dados Teste (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, nb_predict_test)))
print()
#Criando uma Confusion Matrix dos dados de teste
matrix = metrics.confusion_matrix(Y_teste, nb_predict_test, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_teste,nb_predict_test,rownames=['Real'],colnames=['Predito'],margins=True))
print()
#Utilizando o modelo RandomForest
print('\nModelo 2 - RandomForest:\n')
#Criando o modelo
modelo_v2 = RandomForestClassifier(n_estimators=1000)
#modelo_v2 = RandomForestClassifier(n_estimators=1000)
modelo_v2.fit(X_treino, Y_treino.ravel())
#Verificando a exatidão no modelo nos dados de treino
rf_predict_train = modelo_v2.predict(X_treino)
print("Exatidão dos Dados de Treino (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, rf_predict_train)))
print()
#Criando uma Confusion Matrix dos dados de treino
matrix = metrics.confusion_matrix(Y_treino, rf_predict_train, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_treino,rf_predict_train,rownames=['Real'],colnames=['Predito'],margins=True))
print()
#Verificando a exatidão no modelo nos dados de teste
rf_predict_test = modelo_v2.predict(X_teste)
print("\nExatidão dos Dados de Teste (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, rf_predict_test)))
print()
#Criando uma Confusion Matrix dos dados de teste
matrix = metrics.confusion_matrix(Y_teste, rf_predict_test, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_teste,rf_predict_test,rownames=['Real'],colnames=['Predito'],margins=True))
print()
#Utilizando o modelo Regressão Logística
print('\nModelo 3 - Regressão Logística:\n')
#Criando o modelo
modelo_v3 = LogisticRegression(solver='lbfgs', max_iter=10000)
#modelo_v3 = LogisticRegression(solver='lbfgs', max_iter=10000)
modelo_v3.fit(X_treino, Y_treino.ravel())
#Verificando a exatidão no modelo nos dados de treino
lr_predict_train = modelo_v3.predict(X_treino)
print("Exatidão dos Dados de Treino (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_treino, lr_predict_train)))
print()
#Criando uma Confusion Matrix dos dados de treino
matrix = metrics.confusion_matrix(Y_treino, lr_predict_train, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_treino,lr_predict_train,rownames=['Real'],colnames=['Predito'],margins=True))
print()
#Verificando a exatidão no modelo nos dados de teste
lr_predict_test = modelo_v3.predict(X_teste)
print("\nExatidão dos Dados de Teste (Accuracy): {0:.4f}".format(metrics.accuracy_score(Y_teste, lr_predict_test)))
print()
#Criando uma Confusion Matrix dos dados de teste
matrix = metrics.confusion_matrix(Y_teste, lr_predict_test, labels=[1,0])
print('Confusion Matrix : \n', matrix)
print()
tn, fp, fn, tp = matrix.ravel()
print('TP:', tp, 'FN:', fn, 'FP:', fp, 'TN:', tn)
print()
print(pd.crosstab(Y_teste,lr_predict_test,rownames=['Real'],colnames=['Predito'],margins=True))
print()
print('\n FIM DO PROCESSO\n\n')
#
################################# FIM DO PROCESSO #################################
#
