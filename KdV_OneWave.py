#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:04:29 2020
@author: isaac_m

Este programa resuelve la ecuacion diferencial
u_t+(f(u))_x+u_xxx=0
por medio del metodo de Galerkin Discontinuo
con polinomios ortogonales de legendre como funciones base
"""


import numpy as np
import matplotlib.pyplot as plt
from math import pi,cosh,sqrt
import numpy.linalg as lg
xbar=-5
c=2
def f(g,x,j):#para graficar la solucion analitica
    n=len(x)
    F=np.zeros(n)
    F=(c/2)/(np.cosh(0.5*np.sqrt(c)*(x-xbar-c*j*Dt)))**2
    return F
def g(x, centro):#condicion inicial para el PVI
    return (c/2)/np.cosh(0.5*np.sqrt(c)*(x-xbar))**2

#__________________________NUMERO DE CELDAS Y FUNCIONES BASE__________________________________
'''el numero de nodos necesariamente aumenta conforme aunmenta el grado de nuestras funciones base'''
num_celdas=30           #<-numero de celdas para el dominio
num_bases=3             #numero de polinomios de legendre a utilizar
num_nodos=5             #<numero de nodos que se desean para integrar por medio de Cuadraturas de Gauss

#___________________________PARA GRAFICAR_____________________________________________________
Dt=0.0007             #<---------longitud de paso en el tiempo
fps=10                #<---------cuadros por cada segundo de t que queremos para la animacion
salto=int(1/(fps*Dt))            
inf=-10            #<---------valor minimo de x para la grafica
sup=12              #<--------valor maximo de x para la grafica
k=10                 #<-----puntos por celda que quiero que se tomen al momento de graficar
tiempo_inicial=0     #<----tiempo inicial(este no cambia)
tiempo_final=6+salto*Dt   #<--------tiempo final que muestra la animacion
'''
LO SIGUIENTE YA ES EL PROGRAMA QUE FUNCIONA EN BASE A LOS PARAMETROS ANTERIORES
'''
M=num_celdas
N=num_bases
n=k*num_celdas     #<----discretizacion para x
x=np.linspace(inf,sup,n)
# print(x)
t = np.arange(tiempo_inicial, tiempo_final, Dt)
iteraciones=len(t)
# print(t)
def cambio_variable(a,b,t):
    '''
    ENTRADAS toma la variable t en un intervalo (a,b)
    SALIDAS  y regresa un valor x en el intervalo (0,1)
    '''
    X=0.5*(b-a)*t+0.5*(a+b)
    return X
def integra(a,b,num_nodos,f,g,centro):
    '''realiza la integral por medio de cuadratura gaussiana'''
    nodos,pesos=np.polynomial.legendre.leggauss(num_nodos)
    x=cambio_variable(a,b,nodos)
    return  np.dot(pesos,(b-a)*f(x,centro)*g(x,centro)/2)
def valores_por_celda(inf,sup,num_celdas):
    centros=np.zeros((num_celdas,3))
    for p in range(num_celdas):
        for q in range(3):
            centros[p,q]=inf+(sup-inf)*(2*p+q)/(2*num_celdas)
            
    return centros
def func_base_0(x,centro):
    return 1+0*x
def func_base_1(x,centro):
    return 2*(x-centro[1])/(centro[2]-centro[0])
def func_base_2(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return 0.5*(3*E**2-1)
def func_base_3(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return 0.5*(5*E**3-3*E)
def Dfunc_base_0(x,centro):
    return 0+0*x
def Dfunc_base_1(x,centro):
    return 2/(centro[2]-centro[0])+0*x
def Dfunc_base_2(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return (3*2)*E/(centro[2]-centro[0])
def Dfunc_base_3(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return 0.5*(5*3*E**2-3) * 2/(centro[2]-centro[0])
funcion_base_num=[func_base_0,func_base_1,func_base_2,func_base_3]
derivada_base_num=[Dfunc_base_0,Dfunc_base_1,Dfunc_base_2,Dfunc_base_3]
I=valores_por_celda(inf,sup,num_celdas)
#print('valores por celda\n',I)
x=np.linspace(inf,sup,n)
def condiciones_iniciales(g):#correcto
    N=num_bases
    M=num_celdas
    C=np.zeros(N*M)
    celda=0
    while celda<M:
        for i in range(N):
            C[i+celda*N]=integra(I[celda,0],I[celda,2],num_nodos,g,funcion_base_num[i],I[celda])\
                         /integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[i],funcion_base_num[i],I[celda,:])
        celda+=1
    return C
C_I=condiciones_iniciales(g)
def A():#es una matriz diagonal
    A=np.zeros((M*N,M*N))
    for celda in range(M):
        for i in range(N):
            A[i+celda*N,i+celda*N]=integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[i],funcion_base_num[i],I[celda,:])
    return A
def P():
    P=np.zeros((M*N,M*N))
    for celda in range(M):
        for k in range(N):
            for l in range(N):
                P[k+celda*N,l+celda*N]= - integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[l],derivada_base_num[k],I[celda,:])\
                - funcion_base_num[l](I[celda,0],I[celda]) * funcion_base_num[k](I[celda,0],I[celda])
                if celda!=M-1: 
                    P[k+celda*N,l+((celda+1)%M)*N] = funcion_base_num[l](I[celda,2],I[(celda+1)%M]) * funcion_base_num[k](I[celda,2],I[celda])

    return P
def Q():#LISTO
    Q = np.zeros((M*N,M*N))
    for celda in range(M):
        for k in range(N):
            for l in range(N):
                Q[celda*N+k,celda*N+l] = integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[l],derivada_base_num[k],I[celda]) \
                + funcion_base_num[l](I[celda,0],I[celda,:]) *  funcion_base_num[k](I[celda,0],I[celda,:])
                if celda!=M-1:
                    Q[celda*N+k,((celda + 1)%M)*N+l] = - funcion_base_num[l](I[celda,2],I[(celda+1)%M,:]) * funcion_base_num[k](I[celda,2],I[celda,:])
    return Q
def U():
    U = np.zeros((M*N,M*N))
    for celda in range(M):
        for k in range(N):
            for l in range(N):
                U[celda*N+k,celda*N+l] = integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[l],derivada_base_num[k],I[celda,:]) \
                - funcion_base_num[l](I[celda,2],I[celda,:]) * funcion_base_num[k](I[celda,2],I[celda,:])
                if celda!=0:
                    U[celda*N+k,((celda-1)%M)*N+l] = funcion_base_num[l](I[celda,0],I[(celda-1)%M,:]) * funcion_base_num[k](I[celda,0],I[celda,:])
            
    return U

def funcion(u):
    return 3*u**2
def comb_lineal(u,x,celda):
    """
    el vector u es el de coeficientes u(t) en el tiempo t,
    este se va actualizando.
    Al comienzo, este es el vector de condiciones iniciales C_I
    """
    E=0
    for l in range(N):
        E+=u[celda*N+l]*funcion_base_num[l](x,I[celda,:])
    return E 
def f_comb_lineal(u,x,celda):
    """
    el vector u es el de coeficientes u(t) en el tiempo t,
    este se va actualizando.
    Al comienzo, este es el vector de condiciones iniciales C_I
    """
    E=0
    for l in range(N):
        E+=u[celda*N+l]*funcion_base_num[l](x,I[celda,:])
    return funcion(E)

def integrar(a,b,num_nodos,f,g,celda,u):
    '''realiza la integral por medio de cuadratura gaussiana'''
    nodos,pesos=np.polynomial.legendre.leggauss(num_nodos)
    x=cambio_variable(a,b,nodos)
    return  np.dot(pesos,(b-a)*f(u,x,celda)*g(x,I[celda,:])/2)
def F_(u): #LISTO
    INTEGRAL = np.zeros((N*M,1))
    EVALUACION = np.zeros((N*M,1))
    for celda in range(M):
        for k in range(N):
            INTEGRAL[celda*N+k,0] = - integrar(I[celda,0],I[celda,2],num_nodos,f_comb_lineal,derivada_base_num[k],celda,u)
            EVALUACION[celda*N+k,0] = f_comb_lineal(u,I[celda,2],celda) * funcion_base_num[k](I[celda,2],I[celda])
            if celda!=0:
                EVALUACION[celda*N+k,0] += - f_comb_lineal(u,I[celda,0],(celda-1)%M) * funcion_base_num[k](I[celda,0],I[celda])
    return INTEGRAL+EVALUACION
def d_sol(x,t):
    derivadas012=[(-c/2)/(np.cosh(0.5*np.sqrt(c)*(x-xbar-c*t)))**2,
                  (0.5*c**1.5)*(np.tanh(0.5*np.sqrt(c)*(-xbar-c*t+x)))/(np.cosh(0.5*np.sqrt(c)*(xbar+c*t-x)))**2,
                  0.25 * c * (c / (np.cosh(0.5*np.sqrt(c)*(xbar+c*t-x)))**4 )\
                   - 2*c*np.tanh(0.5*np.sqrt(c)*(-xbar-c*t+x))**2/(np.cosh(0.5*np.sqrt(c)*(xbar+c*t-x))**2)  ]
    return derivadas012
def F1(t):#LISTO
    FLUX= np.zeros((M*N,1))
    for k in range(N):
        celda=M-1
        #el primer termino es el flujo de p por la frontera izquierda y el segundo es el flujo de F(u) por la izquierda
        FLUX[celda*N+k,0] = d_sol(I[celda,2],t)[2]*funcion_base_num[k](I[celda,2],I[celda])
        celda=0
        FLUX[celda*N+k,0] = - funcion(d_sol(I[celda,0],t)[0])*funcion_base_num[k](I[celda,0],I[celda])

    return FLUX
def F2(t):
    FLUX= np.zeros((M*N,1))
    for k in range(N):
        celda=M-1
        FLUX[celda*N+k,0] = - d_sol(I[celda,2],t)[1]*funcion_base_num[k](I[celda,2],I[celda])
    return FLUX
def F3(t):#LISTO
    FLUX= np.zeros((M*N,1))
    for k in range(N):
        celda=0
        FLUX[celda*N+k,0] = d_sol(I[celda,0],t)[0]*funcion_base_num[k](I[celda,0],I[celda])
    return FLUX
"""
CON ESTO YA TENEMOS TODAS NUESTRAS FUNCIONES EXPRESADAS COMO MATRICES Y VECTORES
"""
A_i=lg.inv(A())
MP=np.dot(A_i,P())
MQ=np.dot(A_i,Q())
MU=np.dot(A_i,U())
MPMQMU=np.dot(MP,np.dot(MQ,MU))
MPMQ=np.dot(MP,MQ)
def F(t,u):
    MF1 = np.dot(A_i,F1(t))
    MF2 = np.dot(A_i,F2(t))
    MF3 = np.dot(A_i,F3(t))
    MF =  np.dot(A_i,F_(u))
    u_t= - MF - np.dot(MPMQMU,u.reshape(M*N,1)) - np.dot(MPMQ,MF3) + np.dot(MP,MF2)-MF1
    return u_t
def RUNGE_KUTTA_3(N,tiempo_inicial,tiempo_final,F,NumIncognitas,condiciones_iniciales):#correcto
    t=np.linspace(tiempo_inicial,tiempo_final,N)
    h=(tiempo_final-tiempo_inicial)/N
    w=np.zeros((N,NumIncognitas))#vector_de arranque
    w[0,:]=condiciones_iniciales
    i=1
    for k in range(1,N):
        k1=(F(t[k-1],w[k-1,:])).reshape(NumIncognitas,)
        k2=(F(t[k-1]+h/2,w[k-1,:]+(h/2)*k1)).reshape(NumIncognitas,)
        k3=(F(t[k-1]+0.75*h,w[k-1,:]+0.75*h*k2)).reshape(NumIncognitas,)
        w[k,:]=w[k-1,:]+(h/9)*(2*k1+3*k2+4*k3)
        if k==100*i:
            print(100*k/N,"%")
            i+=1
    return w

#comentar estas dos lineas y descomentar la 3ra para usar el txt con los resultados ya guardados
coeficientes=RUNGE_KUTTA_3(iteraciones,tiempo_inicial,tiempo_final,F,num_celdas*num_bases,C_I)
np.savetxt("coeficientes_30c3b5ndt0-0007.txt",coeficientes)
# coeficientes=np.loadtxt("coeficientes_30c3b5ndt0-0007.txt")
#MATRIZ PARA GRAFICAR LAS COMBINACIONES LINEALES DE CADA CELDA    
U_=np.zeros((len(x),len(t)//salto))
for j in range(len(t)//salto):#corre para tiempo
    for l in range(num_celdas):
        centro=I[l,:]
        for i in range(k):#corre para espacio x
            for r in range(num_bases):#este es para que haga la sumatoria
                U_[i+k*l,j]+=coeficientes[j*salto,num_bases*l+r]*funcion_base_num[r](x[i+k*l],centro)
#%%
# plt.ion()#para permitir grafica animada
# plt.figure()
# for i in range(len(t)//salto):
#     plt.clf()
#     plt.xlim()
#     plt.plot(x,U_[:,i],color='red',label='Solucion numerica') #sol numerica
# #---------EL SIGUIENTE CICLO ES PARA MOSTRAR LA DISCONTINUIDAD EN CADA CELDA,
# #---------COMENTAR LA LINEA DE ARRIBA Y DESCOMENTAR ESTE EN CASO DE QUE QUIERA MOSTRARLO DE ESA FORMA
# #    for celda in range (k*M):
# #        plt.plot(x[celda*k:(celda+1)*k],U_[celda*k:(celda+1)*k,i],color='red') #sol numerica
#     plt.plot(x,f(g,x,salto*i),color='blue',linewidth=0.8,label='Solucion Analitica')#sol analitica

#     plt.grid(True)
#     plt.axis((inf,sup,-0.5,1.5))
#     plt.xlabel('x')
#     plt.ylabel('U(x,t)')
# #    plt.title('t=%.1f,instante numero %.2f,%.3f'%(salto*i*Dt,salto*i,U_[0,i]-U_[len(x)-1,i]))
#     plt.title('t=%.1f'%(salto*i*Dt))
#     plt.legend()
#     plt.show()
#     plt.pause(0.01)
    
plt.ion()#para permitir grafica animada donde se note la separacion entre celdas
plt.figure()
color=['Orange','Red','green','Magenta']
for i in range(len(t)//salto):
    plt.clf()
    plt.xlim()
    plt.plot(x,f(g,x,salto*i),color='Blue',label='U')       #graficando la solucion analitica
    for celda in range (k*M):
        plt.plot(x[celda*k:(celda+1)*k],U_[celda*k:(celda+1)*k,i],color=color[celda%len(color)])#graficando la aproximacion
    plt.grid(True)
    plt.axis((inf,sup,-0.5,1.5))
    plt.xlabel('x')
    plt.ylabel('$U(x,t)$')
    # plt.title('t=0')
#    plt.title('t=%.1f,instante numero %.2f,%.3f'%(salto*i*Dt,salto*i,U[0,i]-U[len(x)-1,i]))
    plt.title('t=%.1f'%(salto*i*Dt))
    # plt.legend()
    plt.show()
    plt.pause(0.01)
#%%
''' CALCULANDO LA NORMA DE LA SOLUCION '''
for t_ in range(np.shape(coeficientes)[0]//salto):
    sol=0
    for i in range(M):
        for l in range(N):
            sol+=coeficientes[t_,i*N+l]**2 * integra(I[i,0],I[i,2],num_nodos,funcion_base_num[l],funcion_base_num[l],I[i,:])
    print('en t=%.1f'%(salto*t_*Dt),'la norma de U es',sqrt(sol))
