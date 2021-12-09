#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:17:49 2020

@author: isaac_m
"""

'''
este programa aproxima por medio de Galerkin-Discontinuo la solciÃ³n del PVI
u_t+a*u_x=0
g(x,0)=sen(pi*x)        condicion inicial
f(u)=a*u                condicion de flujo (Upwind)
'''
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def g(x, centro):#condicion inicial para el PVI
    return np.sin(pi*x)



#__________________________NUMERO DE CELDAS Y FUNCIONES BASE____________________________________________________________
'''el numero de nodos necesariamente aumenta conforme aunmenta el grado de nuestras funciones base'''
num_celdas=20         #<---------------------numero de celdas para el dominio
num_bases=2#(por el momento solo he definido la primeras 3 funciones bases, pero se pueden agregar conforme se necesiten)
num_nodos=3        #<---------------numero de nodos que se desean para integrar por medio de Cuadraturas de Gauss



#___________________________PARA GRAFICAR_______________________________________________________________________________

Dt=.02           #<------------------------------------longitud de paso en el tiempo
fps=1               #<---------------cuadros por cada segundo de t que queremos para la animación
salto=int(1/(fps*Dt))    
inf=0             #<------------------------------------valor minimo de x para la grafica(no mover, ademas no creo que sea necesario ya que siempre comenzamos en tiempo 0 'espero')
sup=2              #<------------------------------------valor maximo de x para la grafica
k=10               #<------------------------------------puntos por celda que quiero que se tomen al momento de graficar
tiempo_inicial=0   #<------------------------------------tiempo inicial(este no cambia(no es general por el momento))
tiempo_final=60+salto*Dt     #<------------------------------------tiempo final que muestra la animacion



#_____________________________CONDICIONES PARA LA ECUACION______________________________________________________________
a=2/3              #<------------------------------------la 'a' de la ecuacion u_x+a*u_t=0
cond_flujo=0.0

M=num_celdas
N=num_bases

'''
LO SIGUIENTE YA ES EL PROGRAMA QUE FUNCIONA EN BASE A LOS PARAMETROS ANTERIORES
'''
n=k*num_celdas     #<-----------------------------------discretizacion para x
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
            centros[p,q]=sup*(2*p+q)/(2*num_celdas)
    return centros


def func_base_0(x,centro):
    return 1+0*x
def func_base_1(x,centro):
    return 2*(x-centro[1])/(centro[2]-centro[0])
def func_base_2(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return 0.5*(3*E**2-1)
    # return 1
def Dfunc_base_0(x,centro):
    return 0+0*x
def Dfunc_base_1(x,centro):
    return 2/(centro[2]-centro[0])+0*x
def Dfunc_base_2(x,centro):
    E=2*(x-centro[1])/(centro[2]-centro[0])
    return (3*2)*E/(centro[2]-centro[0])


funcion_base_num=[func_base_0,func_base_1,func_base_2]
derivada_base_num=[Dfunc_base_0,Dfunc_base_1,Dfunc_base_2]
I=valores_por_celda(inf,sup,num_celdas)
# print('valores por celda\n',I)

x=np.linspace(inf,sup,n)
def matriz_pesos_derivada():#correcto
    M=num_celdas
    N=num_bases
    print(N)
    A=np.zeros((M*N,M*N))
    celda=0
    while celda<M:
        for i in range(N):
            A[i+celda*N,i+celda*N]=integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[i],funcion_base_num[i],I[celda,:])
        celda+=1
    return A
A=matriz_pesos_derivada()
print('matriz de la derivada',A)

def matriz_pesos_coeficientes():#correcto
    M=num_celdas
    N=num_bases
    B=np.zeros((M*N,M*N))
    celda=0
    for i in range(N):
        for j in range(N):
            #PARA CONDICIONES PERIODICAS
            B[i,M*N-N+j]=-a*funcion_base_num[j](I[M-1,2],I[M-1,:])*funcion_base_num[i](I[0,0],I[0,:])
    while celda<M:
        for j in range(N):
            for i in range(N):
                B[i+celda*N,j+celda*N]=-a*integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[j],derivada_base_num[i],I[celda,:])\
                +a*funcion_base_num[j](I[celda,2],I[celda,:])*funcion_base_num[i](I[celda,2],I[celda,:])
                if celda>=1:
                    B[i+celda*N,j+celda*N-N]=-a*funcion_base_num[j](I[celda,0],I[celda-1,:])*funcion_base_num[i](I[celda,0],I[celda,:])
                   # print('en la celda',celda,'x_i-1/2=',I[celda-1,0])
        celda+=1
    print("B=\n",B)
    return B
B=matriz_pesos_coeficientes()
#%%
def condiciones_iniciales(g):#correcto
    N=num_bases
    M=num_celdas
    C=np.zeros(N*M)
    celda=0
    while celda<M:
        for i in range(N):
            C[i+celda*N]=integra(I[celda,0],I[celda,2],num_nodos,g,funcion_base_num[i],I[celda,:])/integra(I[celda,0],I[celda,2],num_nodos,funcion_base_num[i],funcion_base_num[i],I[celda,:])
        celda+=1
    return C
CI=condiciones_iniciales(g)
print('condiciones iniciales:\n',CI)
print(num_celdas)
print(num_bases)
num_coeficientes=num_celdas*num_bases

A_inversa=np.linalg.solve(A,np.eye(len(A)))
def F(t,vector_de_arranque):#correcto
    comienzo=-np.dot(np.dot(A_inversa,B),vector_de_arranque.reshape(np.shape(vector_de_arranque)[0],1))
    return comienzo
    #%%
def RUNGE_KUTTA_3(N,tiempo_inicial,tiempo_final,F,NumIncognitas,condiciones_iniciales):#correcto
    t=np.linspace(tiempo_inicial,tiempo_final,N)
    h=(tiempo_final-tiempo_inicial)/N
    w=np.zeros((N,NumIncognitas))#vector_de arranque
    k1=np.zeros((N,NumIncognitas))
    k2=np.zeros((N,NumIncognitas))
    k3=np.zeros((N,NumIncognitas))
    w[0,:]=condiciones_iniciales
    for k in range(1,N):
        k1[k,:]=h*F(t[k-1],w[k-1,:]).reshape(NumIncognitas,)
        k2[k,:]=h*F(t[k-1]+h/2,w[k-1,:]+(h/2)*k1[k,:]).reshape(NumIncognitas,)
        k3[k,:]=h*F(t[k-1]+h,w[k-1,:]+h*k1[k,:]).reshape(NumIncognitas,)
        w[k,:]=w[k-1,:]+(1/6)*(k1[k,:]+4*k2[k,:]+k3[k,:])
#    print(F(t[k-1],w[k-1,:]))
    return w
import numpy.linalg as lg
def Crank_Nicolson(N,tiempo_inicial,tiempo_final,F,NumIncognitas,condiciones_iniciales):
#    t=np.linspace(tiempo_inicial,tiempo_final,N)
    h=(tiempo_final-tiempo_inicial)/N
    w=np.zeros((N,NumIncognitas))
    w[0,:]=condiciones_iniciales
    i=1
    for k in range(1,N):
        C=-np.dot(A_inversa,B)
        w[k,:]=np.dot(np.dot(lg.inv(np.eye(len(C))-0.5*h*C),np.eye(len(C))+0.5*h*C) , w[k-1,:].reshape(len(w[k-1,:]),))
        if k==100*i:
            print(100*k/N,"%")
            i+=1
    return w
#coeficientes=RUNGE_KUTTA_3(iteraciones,tiempo_inicial,tiempo_final,F,num_celdas*num_bases,CI)
coeficientes=Crank_Nicolson(iteraciones,tiempo_inicial,tiempo_final,F,num_celdas*num_bases,CI)
#%%
 # print(coeficientes)
#for i in range(num_celdas*num_bases):
#    APROX=coeficientes[:,i]#la i-th columna nos da el valor de C^i_j(t)
#                             #las j-th fila nos da el valor de C^i_j(j*dt)
#     # print('aprox',APROX)
#    plt.plot(t,APROX,color='red')
#    plt.show()
U_=np.zeros((len(x),len(t)//salto))
for j in range(len(t)//salto):#corre para tiempo
    for l in range(num_celdas):
        centro=I[l,:]
        for i in range(k):#corre para espacio x
            for r in range(num_bases):#este es para que haga la sumatoria
                U_[i+k*l,j]+=coeficientes[j*salto,num_bases*l+r]*funcion_base_num[r](x[i+k*l],centro)

def f(g,x,j):
    n=len(x)
    F=np.zeros(n)
    for i in range(n):
#        if x[i]>=0.0+a*j*Dt:
        F[i]=g(x[i]-a*j*Dt,None)
    return F
#%%
# plt.ion()#para permitir grafica animada
# plt.figure()
# for i in range(len(t)//salto):
#     plt.clf()
#     # plt.plot(xv,SOL[10*i,:],color='green')
#     plt.plot(x,U[:,salto*i],color='red',label='Solución Numérica') #graficando la aproximacion
#     plt.plot(x,f(g,x,salto*i),color='blue',linewidth=0.75,label='Solución Analítica')#graficando la solucion analitica
#     plt.grid(True)
#     plt.axis((inf,sup,-1.5,1.5))
#     plt.xlabel('x')
#     plt.ylabel('U(x,t)')
#     # plt.title('t=0')
#     plt.title('t=%.1f,instante numero %.2f'%(salto*i*Dt,salto*i))
# #    plt.legend()
#     plt.show()
#     plt.pause(0.001)
plt.ion()#para permitir grafica animada
plt.figure()
color=['Orange','Red','green','Magenta']
for i in range(len(t)//salto):
    plt.clf()
    plt.xlim()
    plt.plot(x,f(g,x,salto*i),color='Blue',label='U')       #graficando la solucion analitica
    for celda in range (k*M):
        plt.plot(x[celda*k:(celda+1)*k],U_[celda*k:(celda+1)*k,i],color=color[celda%len(color)])#graficando la aproximacion
    plt.grid(True)
    plt.axis((inf,sup,-1.1,1.1))
    plt.xlabel('x')
    plt.ylabel('$U(x,t)$')
    # plt.title('t=0')
#    plt.title('t=%.1f,instante numero %.2f,%.3f'%(salto*i*Dt,salto*i,U[0,i]-U[len(x)-1,i]))
    plt.title('t=%.1f'%(salto*i*Dt))
#    plt.legend()
    plt.show()
    plt.pause(.1)
#%%
    
''' CALCULANDO LA NORMA DE LA SOLUCION '''
for t_ in range(np.shape(coeficientes)[0]//salto):
    sol=0
    for i in range(M):
        for l in range(N):
            sol+=coeficientes[t_,i*N+l]**2 * integra(I[i,0],I[i,2],num_nodos,funcion_base_num[l],funcion_base_num[l],I[i,:])
    print('en t=%.1f'%(salto*t_*Dt),'la norma de U es',np.sqrt(sol))
#%%
""" Calculando el error ||u_h - u|| EN EL TIEMPO"""
def solucion(x,t):
    return np.sin(pi*(x-a*t))
def sol_num(x,t_,celda):
    u_h=0
    for l in range(N):
        u_h += coeficientes[t_,celda*N+l] * funcion_base_num[l](x,I[celda,:])
    return u_h
def error(x,t_,celda):
    return abs(solucion(x,t_*Dt*salto) - sol_num(x,t_*salto,celda))
def ERROR(u,u_h,t,celda,num_nodos):
    '''realiza la integral por medio de cuadratura gaussiana'''
    nodos,pesos=np.polynomial.legendre.leggauss(num_nodos)
    a=I[celda,0]
    b=I[celda,-1]
    x=cambio_variable(a,b,nodos)
#    print(x)
    integral=np.dot(pesos,(b-a)*(error(x,t,celda))**2  /2)
    return  integral

E_a=np.zeros(len(t)//salto)
for t_ in range(0,len(t)//salto):
    print(t_)
    for celda in range(M):
        E_a[t_] += ERROR(solucion,sol_num,t_,celda,num_nodos)
#        print(E_a[t_])
    print('en t=%.1f'%(t_*salto*Dt),'el ERROR es',np.sqrt(E_a[t_]))
plt.figure()
T=np.linspace(t[0],60,len(t)//salto)
plt.plot(T,np.sqrt(E_a),'.')
plt.xlabel('t')
plt.ylabel('$E_a=||u_h-u||$')
plt.grid(True)
plt.show()
