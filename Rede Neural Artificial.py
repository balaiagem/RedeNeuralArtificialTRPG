    from os import close
    import numpy as np
    from numpy.lib import append
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn import datasets
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from pybrain3.tools.shortcuts import buildNetwork
    from pybrain3.datasets import SupervisedDataSet
    from pybrain3.supervised.trainers import BackpropTrainer
    from pybrain3.structure.modules import SoftmaxLayer
    from pybrain3.structure.modules import SigmoidLayer
    from pybrain3.tools.validation import CrossValidator, ModuleValidator
    import matplotlib.pyplot as plt
    from datetime import datetime
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    
    #Função utilizada para criação da Rede Neural Artificial
    #"rede" é a Multilayer Perceptron e "base" é o conjundo de entradas e saídas.   
    rede = buildNetwork(9, 100,100, 1)
    base = SupervisedDataSet(9, 1) 




    #Função responsável por ler o arquivo com a base de dados escrita
    #Depois a base de dados é lida e as entradas são separadas das saídas
    #Por fim, uma amostra é adicionada no dataset
    arquivo = open('base.txt', 'r')
    arquivo.seek(0, 0)

    for linha in arquivo.readlines():
        l = [float(x) for x in linha.strip().split(',') if x != '']
        indata = l[:9]
        outdata = l[9:]
        base.addSample(indata, outdata)


    #Função de treinar a rede neural artificial
    #As variaveis arraydeerros e arraydetentativa servem para armazenar as iterações que serão projetadas em gráfico
    #A array aprendizado armazena os erros de cada iteração durante o treinamento.
    arraydeerros = []
    arraydetentativa = []
    treinamento = BackpropTrainer(rede, dataset=base, learningrate=0.01)
    error = 2
    iteration = 0 
    aprendizado=[]
    while error > 0.001:
        error = treinamento.train()
        iteration += 1
        arraydeerros.append(error)
        arraydetentativa.append(iteration)
        aprendizado.append(error)

    #Gerar gráfico de aprendizado Tentativa x Erro
    plt.plot(arraydetentativa,arraydeerros)
    plt.title('Gráfico de aprendizado de máquina')
    plt.xlabel('Nº da Tentativa')
    plt.ylabel('Valor do Erro')
    plt.savefig('grafico.jpg', format='jpg')
    plt.show()


    #a array entradas e saídas são responsável por armazenar todas entradas e saídas, respectivamente
    #a array pred armazena o resultado do teste realizado, chamado de predicao
    #por fim, é calculada a acuracia da rede, utilizando os resultados desejados comparados com os preditos
    basedetestes = open('testes.txt', 'r')
    entradas = []
    saidas = []
    pred = []
    basedetestes.seek(0, 0)
    for linha in basedetestes.readlines():
        l = [int(x) for x in linha.strip().split(',') if x != '']
        indata = l[:9]
        outdata = l[9:]
        entradas.extend([indata])
        saidas.extend([outdata])
        pred.extend([int(round(float(rede.activate(indata))))])

    accuracy_score(saidas,pred)

