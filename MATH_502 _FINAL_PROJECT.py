#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib.pyplot import *
from numpy import *
from scipy.integrate import odeint


# $$$$

# # A  SIMPLE ROSS-MACDONALD MATHEMATICAL MODEL OF MOSQUITO-BORNE PATHOGEN TRANSMISSION#

# # Darlington Christopher and Ta-En(Darwin) Lo#

# # April 13, 2019#

# $$$$
# $$$$
# $$$$

# # ABSTRACT#

# A mathematical model called the Ross-Macdonald Model is introduced, the model involves the concept of
# differential equations and although altered for many different reasons, it is most useful in the control of the spread of mosquitoes and consequently the mitigation of mosquito-borne diseases. The Ross-Macdonald Model has played a critical part in the research of mosquito-borne pathogen transmission as well as the development of strategies for mosquito-borne disease prevention.
# 
# Malaria is an intermittent and remittent fever caused by a protozoan parasite, that invades the red blood cells.
# The parasite is transmitted by mosquitoes in many tropical and subtropical regions for example in large areas of Africa and Asia. It causes headache, sweating, fever and in worst cases, death! [[2]].
# 
# [2]: https://www.slideshare.net/BenWielgosz/introduction-to-the-agroecology-of-malaria
# 
# Mosquitoes transmit the pathogens that cause malaria, filariasis, dengue, yellow fever, West Nile fever, Rift
# Valley fever, and dozens of other infectious diseases of humans, domestic animals, and wildlife [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/.
# 
# In this project, we analysed different values of the parameters the linear first order ordinary differential equations in the model depended on, in order to see their effects on the solution of the Ross-Macdonald Model. We also used **odeint** differential equations solver to compare solutions. **The stability of the Disease-Free Equilibrium (DFE)**, where the number of infected mosquitoes and infected humans equal $0$, was also analysed.

# $$$$
# $$$$
# $$$$

# # INTRODUCTION#

# Physicians and scholars have always been suspicious of mosquitoes of transmitting pathogens, but the mosquito hypothesis was neither formally tested nor widely accepted until the late 19th century. **Patrick Manson**, working in China in 1877, was the first to formally demonstrate that mosquitoes transmit a blood-borne pathogen; the filarial worm *Wuchereria bancrofti* was initially isolated from mosquitoes that had fed on his gardener. **Charles Laveran** observed malaria parasites during 1880 using a light microscope, and several people independently formed the hypothesis that malaria parasites could be transmitted by mosquitoes. **Ronald Ross** discussed malaria with Manson while in the UK, but conducted his research while serving in a military post in India, and in 1897 he demonstrated that mosquitoes transmit malaria parasites. Ross argued that mosquito population densities could be reduced through larval control in combination of other measures to prevent mosquito-transmitted diseases. He became a force to be reckoned with in public health and economic benefits of control in publications, speeches, and debates [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/.
# 
# The simplest models of malaria called the Ross-Macdonald models, date to the early $20^{th}$ century, named after **Ronald Ross** and **George Macdonald**. These models simplify the complex interactions amongst human, the female Anopheles mosquito and plasmodium into coupled **differential equations** that specify the changes of two variables $I_h(t)$ and $I_m(t)$ with time. These models are all based on an agreed set of simplifying assumptions [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html.
# 
# The mathematical model presented here is just one of many models for the dynamics and control of mosquito-transmitted pathogens that also includes epidemiological and entomological concepts as well as standards for measuring transmission of mosquito-borne pathogens. 
# Precisely, Ross-Macdonald models assume that transmission of plasmodium from female mosquitoes to humans, and from humans to mosquitoes depends on the **numbers of susceptible and infected individuals
# in the population of the species**.
# 
# A Ross-Macdonald model is based on a simplified process-based quantitative description of the pathogen's life cycle in four steps:
# 
# - The pathogen is passed from an infected mosquito to a vertebrate host during blood feeding 
# - The pathogen infects and then multiplies in the vertebrate host, reaching sufficiently high levels in peripheral blood, capable of infecting a new mosquito
# - A susceptible mosquito digests the pathogen from the infected vertebrate host during feeding; and
# - The pathogen develops in the mosquito to a point that it is in the salivary glands or mouth parts and thus ready to be transmitted during a subsequent bite on a susceptible vertebrate host.
# 
# 
# The infection dynamics in the mosquito are based on a simplified description of the mosquito cycle of blood feeding and egg-laying. The models differ in the way they **implement latency** in the mosquito, but there is generally in most cases, an agreed set of simplifying assumptions about the transmission dynamics as follows:
# 
# - The mosquito bites are distributed randomly and evenly among vertebrate host populations
# - Populations are closed to birth or migration
# - There are many more humans than infectious bites
# - There is one vertebrate host (usually humans)
# - Human infections are simple and clear at a constant per-capita rate
# - Hosts become immediately susceptible to infection after recovery (at rate r), gives an exponential distribution of infected humans
# - The ratio of mosquitoes to humans is constant
# - Female mosquito mortality is independent of age, so that the mosquito lifespan is exponentially distributed
# - The pathogen latent period in mosquitoes is constant
# - There is only one mosquito vector species (female Anopheles)
# - There is a constant fraction of mosquitoes blood feed on the pathogen's host [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/

# $$$$
# $$$$
# $$$$

# # THE SIMPLE MODEL#

# $a$,$b$,and $c$ represent a relationship between transmission and biting by female Anopheles mosquitoes - how transfer occurs, the probabilities of parasite transfer from vectors to humans and humans to vectors!
# $$$$
# The Product $abI_m(1-I_h)$ is the mosquito production of newly infected humans.
# The Product $acI_h(1-I_m)$ is the human production of newly parasitized mosquitoes [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html.
# 
# Where $rI_h$ can be thought of as the rate of recovery of infected humans and $\mu_2I_m$ can be thought of as the death rate of infected mosquitoes.
# 
# $\frac{dI_h}{dt}$ is the rate of change of the number of infected humans with time and $ \frac{dI_m}{dt}$ is the rate of change of the number infected mosquitoes with time.
# 
# The model consists of a system of two linear ODEs as follows:
# $$$$
# 
# $\frac{dI_h}{dt} = abmI_m(1-I_h) - rI_h$   $\dots$ (1)
# 
# $ \frac{dI_m}{dt} = acI_h(1-I_m) - \mu_2I_m$  $\dots$ (2)
# $$$$
# **The parameters**
# 
# - $I_h$ = the number of infected humans measured in humans
# - $I_m$ = the number of infected mosquitoes measured in mosquitoes
# - $a$ = the mosquito biting rate measured in bites per mosquito per time 
# - $b$ = mosquito to human transmission probability measured in per bite per mosquito
# - $c$ = human to mosquito transmission probability measured in per bite per human
# - $m$ = the number of mosquitoes per human measured in total number of mosquitoes per total number of humans
# - $r$ = the human recovery rate measured in per unit of time 
# - $\mu_2$ = the mosquito death rate also measured in per unit of time [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/
# 
# $R_0 =\frac{a^2bcm}{r\mu_2}$ is the basic transmission rate of the malaria parasite between humans and mosquitoes [[2]].
# 
# [2]: https://www.slideshare.net/BenWielgosz/introduction-to-the-agroecology-of-malaria
# 
# With the above parameters, it is reasonable to say that the average duration of human infection or the mean recovery time in humans, $D_H$ is $\frac{1}{r}$ and the average duration of mosquito infection or the mean longevity of mosquitoes, $D_M$ is $\frac{1}{\mu_2}$ [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html..

# $$$$
# $$$$
# $$$$

# # THE LINEARIZATION OF THE MODEL#
# 

# $$J= 
# \begin{bmatrix}
# -abmI_m -r & abm(1-I_h)  \\                         
# ac(1-I_m) & -acIh -\mu_2\\
# \end{bmatrix}  \dots (3)$$ [[5]].
# 
# [5]: https://arxiv.org/pdf/1803.05291.pdf 
# 
# At the Disease Free Equilibrium ($DFE$), where $(I_h,I_m) = (0,0)$, The matrix above becomes:
# 
# $$J|_{DFE}= 
# \begin{bmatrix}
# -r & abm \\                         
# ac &  -\mu_2\\
# \end{bmatrix}  \dots (4) $$ [[4]].
# 
# [4]: http://www.tnstate.edu/mathematics/mathreu/filesreu/GroupProjectSIR.pdf
# 
# 
# $$J|_{DFE}-\lambda I = \begin{bmatrix}
# -r-\lambda & abm \\                         
# ac &  -\mu_2-\lambda\\
# \end{bmatrix}  \dots (5) $$ 
# 
# To solve for the eigenvalues and eigenvectors, we find the values of $\lambda$ that makes the determinant = 0
# 
# $$det(J|_{DFE}-\lambda I)=\begin{vmatrix} -r & abm \\ ac & -\mu_2 \end{vmatrix} = 0 \implies \lambda ^2 + (r+\mu_2)\lambda +(r\mu_2 - a^2bcm) = 0 \dots (6)
# $$ 
# 
# **This is called the characteristic equation.**

# $$$$

# **SEVEN DIFFERENT CASES WERE CONSIDERED**

# $$$$

# **CASE ONE (THE REFERENCE CASE FOR COMPARISON)**
# 
# 
# 

# $$$$

# $a=0.5, b=0.33, c=0.33, m=100, r =2, \mu_2 =5$

# $$$$

# From the characteristic equation, plugging in these values yields:
# 
# $$\lambda ^2 + 7\lambda +7.2775 = 0 \dots (7)$$
# $$$$
# This takes the form of $\lambda ^2 - tr(J|_{DFE}) + det(J|_{DFE})$ [[5]].
# 
# [5]: https://arxiv.org/pdf/1803.05291.pdf 
# $$$$
# By using the quadratic formula:\begin{array}{*{20}c} {\lambda_{1,2} = \frac{{ - b \pm \sqrt {b^2 - 4ac} }}{{2a}}} & {{\rm{where}}} & {a\lambda^2 + b\lambda + c = 0} \dots (8)\\ \end{array}  we get the following eigenvalues:
# 
# 
# $\lambda_1 = -1.27$ and $\lambda_2 = -5.73$, substituting these values into the matrix $$J|_{DFE}-\lambda I = \begin{bmatrix}
# -2-\lambda & 16.5 \\                         
# 0.165 &  -5-\lambda\\
# \end{bmatrix}  \dots (9) $$ yields the corresponding eigenvectors:
# 
# $$v_1 =\begin{bmatrix}
# 22.6055 \\
# 1 
# \end{bmatrix}  $$ and
# 
# 
# 
# $$v_2 =\begin{bmatrix}
# -4.42370\\
# 1 
# \end{bmatrix} $$
# 
# 
# So that 
# 
# $$U(t) = [I_h(t), I_m(t)]^T = c_1exp(-1.27t)\begin{bmatrix}
# 22.6055\\
# 1 
# \end{bmatrix} + c_2exp(-5.73t)\begin{bmatrix}
# -4.42370\\
# 1 
# \end{bmatrix} \dots (10) $$ 
# 
# Using the initial condition $U(0) = [0.1,0.1]^T$ for convenience for all cases, where this means the initial number of infected humans $I_h = 0.1$ and the initial number of infected mosquitoes $I_m =0.1$. Two systems of equations in $c_1$ & $c_2$ arise, they are as follows:
# 
# $$c_1 + c_2 = 0.1$$
# $$22.6055c_1 -4.42370c_2= 0.1  \dots (11) $$
# 
# **Using Cramer's rule to solve the above system:**
# $$$$
# 
# 
# $$ \begin{bmatrix}
#  22.6055 & -4.42370  \\
# 1 & 1 
# \end{bmatrix}  \begin{bmatrix}
# c_1 \\
# c_2
# \end{bmatrix}  = \begin{bmatrix}\\
# 0.1 \\
# 0.1 
# \end{bmatrix}  $$ 
# 
# $$$$
# 
# 
# $$c_1 = \frac{\begin{vmatrix}
# 0.1 & -4.42370 \\
# 0.1 & 1 
# \end{vmatrix}}  {\begin{vmatrix}
#  22.6055 & -4.42370 \\
# 1 & 1
# \end{vmatrix}}  = 0.1822$$
# 
# $$$$
# 
# $$c_2 = \frac{\begin{vmatrix}
# 22.6055 & 0.1 \\
# 1 & 0.1 
# \end{vmatrix}}  {\begin{vmatrix}
#  22.6055 & -4.42370\\
# 1 & 1
# \end{vmatrix}} = 0.8178 $$
# 
# 
# This gives: $$U_1(t) = I_{h}(t) = 0.4544exp(-1.27t) -0.3536exp(-5.73t) \dots(12) $$
# 
# $$U_2(t) = I_{m}(t) = 0.0201exp(-1.27t) + 0.07993exp(-5.73t) \dots(13)$$
# 

# $$$$

# In[22]:


t=linspace(0,1,50)


# In[89]:


Ihm = 0.4544*exp(-1.27*t) - 0.3536*exp(-5.73*t)
Imm = 0.0201*exp(-1.27*t) + 0.07993*exp(-5.73*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# From the above image, we see that with an initial value of 0.1 infected mosquito, the number of infected mosquitoes declines, tending to 0.01. It can also be observed that with an initial value of 0.1 infected humans, the number increases steadily to about 0.24 and then falls sharply to about 0.12.

# In[53]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# $$$$

# In[54]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.33
m= 100
r= 2
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[90]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[115]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='best');


# $$$$

# $$$$

# $$$$

# **CASE TW0 (THE EFFECT OF A CHANGE IN 'a')**

# $a=0.8, b=0.33, c=0.33, m=100, r =2, \mu_2 =5$

# 
# This gives: $$U_1(t) = I_h(t) = 0.5086exp(-0.4636t) -0.4097exp(-6.5364t) \dots(14) $$
# 
# $$U_2(t) = I_m(t) = 0.0296exp(-0.4636t) + 0.0704exp(-6.5364t) \dots(15)$$

# In[116]:


Ihm = 0.5086*exp(-0.4636*t) - 0.4097*exp(-6.5364*t)
Imm = 0.0296*exp(-0.4636*t) + 0.0704*exp(-6.5364*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# As expected, an increase in $a$, the mosquito biting rate will tend to increase the number of infected humans and consequently have a noticeable effect on the number of infected mosquitoes in the population. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.38 and then declines to about 0.32 (a 220% final increase) just by an increase in the biting rate by 60%. $I_m$ tends to about 0.02.

# In[117]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[118]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.8
b= 0.33
c= 0.33
m= 100
r= 2
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[119]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[123]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='lower right');


# $$$$

# $$$$

# **CASE THREE (THE EFFECT OF A CHANGE IN 'b')**

# $a=0.5, b=0.7, c=0.33, m=100, r =2, \mu_2 =5$

# This gives: $$U_1(t) = I_h(t) = 0.6933exp(-0.6672t) -0.5945exp(-6.3328t) \dots(16) $$
# 
# $$U_2(t) = I_m(t) = 0.0264exp(-0.6672t) + 0.0736exp(-6.3328t) \dots(17)$$

# In[124]:


Ihm = 0.6933*exp(-0.6672*t) - 0.5945*exp(-6.3328*t)
Imm = 0.0264*exp(-0.6672*t) + 0.0736*exp(-6.3328*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time');
ylabel('Ih and Im')
legend(loc='best');
show()


# As expected, an increase in $b$, the mosquito to human transmission probability tends to increase the number of infected humans.which makes sense! A higher mosquito to human transmission probability means a higher likelihood for transmission. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.48 and then declines to about 0.36 (a 260% final increase) just by an increase in the mosquito to human transmission probability by 112%. $I_m$ tends to about 0.01.

# In[125]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[126]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.7
c= 0.33
m= 100
r= 2
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[127]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[129]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='lower right');


# $$$$

# $$$$

# **CASE FOUR (THE EFFECT OF A CHANGE IN 'c')**

# $a=0.5, b=0.33, c=0.2, m=100, r =2, \mu_2 =5$

# This gives: $$U_1(t) = I_h(t) = 0.5073exp(-1.5252t) - 0.4036exp(-5.4748t) \dots(18) $$
# 
# $$U_2(t) = I_m(t) = 0.0146exp(-1.5252t) + 0.085exp(-5.4748t) \dots(19)$$

# In[130]:


Ihm = 0.5073*exp(-1.5252*t) - 0.4036*exp(-5.4748*t)
Imm = 0.0146*exp(-1.5252*t) + 0.085*exp(-5.4748*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# Again as expected, a decrease in $c$, the human to mosquito transmission probability should mean that there are less infected mosquitoes and also less infected humans. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.24 and then declines to about 0.11 (a 10% final increase) just by a decrease in the mosquito to human transmission probability by 39.39%. $I_m$ tends to about 0.01.

# In[131]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[132]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.2
m= 100
r= 2
u= 5
z_init=[0.1,0.1]
t=linspace(0,1,50)
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[133]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[134]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='best');


# $$$$

# $$$$

# **CASE FIVE (THE EFFECT OF A CHANGE IN '$\mu_2$')**

# $a=0.5, b=0.33, c=0.33, m=100, r =2, \mu_2 =20$

# This gives: $$U_1(t) = I_h(t) = 0.1870exp(-1.85t) -0.0894exp(-20.15t) \dots(20) $$
# 
# $$U_2(t) = I_m(t) = 0.0017exp(-1.85t) + 0.0983exp(-20.15t) \dots(21)$$

# In[135]:


Ihm = 0.1870*exp(-1.85*t) - 0.0894*exp(-20.15*t)
Imm = 0.0017*exp(-1.85*t) + 0.0983*exp(-20.15*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# Again as expected, an increase in $\mu_2$, the mosquito death rate should mean that there are less infected mosquitoes and also less infected humans in the population, which is quite intuitive. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.14 and then declines to about 0.03 (a 70% final decrease) by an increase in the mosquito death rate by 300%. $I_m$ tends to 0 rapidly. It is interesting to note that the shape of the red curve is narrower here compared to the ones above. The decline of $I_h$ starts at about $t = 0.16$ as opposed to roughly $t = 0.3$ for the others. This suggests that coming up with solutions that increase the death rate of mosquitoes could be a better strategy for malaria mitigation than the ones above. Think about it, it makes perfect sense!

# In[136]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[137]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.33
m= 100
r= 2
u= 20
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[138]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[139]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='best');


# $$$$

# $$$$

# **CASE SIX (THE EFFECT OF A CHANGE IN '$r$')**

# $a=0.5, b=0.33, c=0.33, m=100, r =20, \mu_2 =5$

# This gives: $$U_1(t) = I_h(t) = 0.1087exp(-1.85t) -0.0092exp(-20.15t) \dots(22) $$
# 
# $$U_2(t) = I_m(t) = 0.100exp(-1.85t) + 0.0001exp(-20.15t) \dots(23)$$

# In[145]:


Ihm = 0.1087*exp(-1.85*t) - 0.0092*exp(-20.15*t)
Imm = 0.100*exp(-1.85*t)  + 0.0001*exp(-20.15*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# An increase in $r$, the human recovery rate means that there are less infected humans and also less infected mosquitoes in the population, which is quite intuitive. With the initial values of $I_h$ and $I_m$, $I_h$ declines to about 0.02 (an 80% final decrease) by an increase in the human recovery rate by 900%. $I_m$ tends to 0.02. It is interesting to note that the shape of the red curve is narrowest here compared to others. The decline of $I_h$ starts at about $t = 0.16$ as opposed to roughly $t = 0.3$ for most of the cases. 
# Comparing this to case 5 above, a $900%$ increase in $r$ led to an $80%$ final decrease in $I_h$ while a $300%$ increase in $\mu_2$ led to a $70%$ final decrease in $I_h$, so, yet again! a strategy aimed at the increase in $\mu_2$ seems to be the best strategy still.

# In[146]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[150]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.33
m= 100
r= 20
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[148]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[149]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='best');


# $$$$

# $$$$

# $$$$

# **CASE SEVEN (JUST A CURIOUS ANALYSIS OF THE EFFECTS OF CHANGES IN 'm' IF POPULATIONS WERE NOT CLOSED)**

# - $a=0.5, b=0.33, c=0.33, m=200, r =2, \mu_2 =5$

# This gives: $$U_1(t) = I_h(t) = 0.6709exp(-0.7260t) -0.5721exp(-6.2740t) \dots(24) $$
# 
# $$U_2(t) = I_m(t) = 0.0259exp(-0.7260t) + 0.0741exp(-6.2740t) \dots(25)$$

# In[151]:


Ihm = 0.6709*exp(-0.7260*t) - 0.5721*exp(-6.2740*t)
Imm = 0.0259*exp(-0.7260*t) + 0.0741*exp(-6.2740*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time ');
ylabel('Ih and Im')
legend(loc='best');
show()


# An increase in $m$, the number of mosquitoes per human increases the number of infected humans. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.47 and then declines to about 0.32 (a 220% final increase) by an increase in the number of mosquitoes per human by 100%. $I_m$ tends to 0.01.

# In[152]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[153]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.33
m= 200
r= 2
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[154]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time');
ylabel('Ih and Im ');


# In[156]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='lower right');


# $$$$

# $$$$

# - $a=0.5, b=0.33, c=0.33, m=50, r =2, \mu_2 =5$

# This gives: $$U_1(t) = I_h(t) = 0.3071exp(-1.5997t) -0.2065exp(-5.4003t) \dots(26) $$
# 
# $$U_2(t) = I_m(t) = 0.0149exp(-1.5997t) + 0.0851exp(-5.4003t) \dots(27)$$

# In[157]:


Ihm = 0.3071*exp(-1.5997*t) - 0.2065*exp(-5.4003*t)
Imm = 0.0149*exp(-1.5997*t) + 0.0851*exp(-5.4003*t)
plot(t,Ihm,'r-',linewidth=2,label='Ih');
plot(t,Imm,'b-',linewidth=2,label='Im');
xlabel('time');
ylabel('Ih and Im')
legend(loc='best');
show()


# A decrease in $m$, the number of mosquitoes per human decreases the number of infected humans. With the initial values of $I_h$ and $I_m$, $I_h$ reaches about 0.15 and then declines to about 0.06 (a 40% final decrease) by a decrease in the number of mosquitoes per human by 100%. $I_m$ tends to 0.01.

# In[158]:


plot(Ihm,Imm);
xlabel('Ih');
ylabel('Im');


# In[159]:


def rhs(z,t,a,b,m,r,c,u):
    Ih,Im = z
    return (a*b*m*z[1])*(1-z[0])- r*z[0], (a*c*z[0])*(1-z[1])- u*z[1]

a= 0.5
b= 0.33
c= 0.33
m= 50
r= 2
u= 5
z_init=[0.1,0.1]
z=odeint(rhs,z_init,t,args=(a,b,m,r,c,u))
Ih,Im =z[:,0],z[:,1]
ylabel('Im')
xlabel('Ih')
plot(z[:,0],z[:,1]);


# In[160]:


plot(t,Ih,'r-',label='Ih');
plot(t,Im,'b-',label='Im');
legend(loc='best');
xlabel('time ');
ylabel('Ih and Im ');


# In[161]:


plot(t,Imm,'r-',label='Imm')
plot(t,Im,'b-',label='Im')
plot(t,Ihm,'y-',label='Ihm')
plot(t,Ih,'g-',label='Ih');
xlabel('time ');
ylabel('Ihm,Ih,Imm and Im')
legend(loc='best');


# $$$$

# $$$$

# # SOME LIMITATIONS OF THE MODEL#

# There were a lot of biological processes omitted in the model, for instance, parasites usually undergo a period of development within the mosquito's gut before entering her salivary glands. So, a latent period, t, exists! a period by which the mosquito is infected but not yet infectious. Therefore, there is a considerable probablity of an infected mosquito's dying before it can pass the disease back to a human host, which varies to an extent with the underlying set of assumptions in the model.
# 
# Differences in changes could arise because of differences in the way the vector behaves, ecology and capability, differences in the dynamics of infection, and immunity in vertebrate hosts. 
# 
# The limitations of acquiring information about transmission to apply the model in context and questions about its relevance, remain as apt as ever. Quantitative tests of the theory continue to present large problems yet to be solved. In particular, fluctuations in mosquito populations are extremely difficult to predict over time and space, and important sources of heterogeneity and the spatial as well as temporal levels of transmission remain incorrectly defined [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/.  
# 
# This model also doesn't incorporate the fact that mosquitoes die soon, after a blood meal if certain protein components are disrupted, as discovered by a team of biochemists at the University of Arizona [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html.

# $$$$

# $$$$

# # CONCLUSION#

# From the above results, we notice that the **eigenvalues** are all **negative and real**, thus by theorem, the **Disease-Free Equilibrium (DFE)** which is $(I_h,I_m) = (0,0)$ is **aymptotically stable**. The infectious mosquitoes population $I_m$ was downward sloping throughout for all cases analysed, this could be explained by the relatively short life-span of mosquitoes for the most part. They could die-off before they could bite the next human.
# $$$$
# The Ross-Macdonald model presented here is quite powerful, despite its many simplifying assumptions. The Ross-Macdonald model has grown into a theory and not just a mathematical model of transmission. It could help in understanding some set of inter-connected empirical phenomena linked to mosquito-borne pathogen transmission. The theory includes the following: 
# - A dynamic model of malaria transmission that has been analyzed extensively; 
# - a set of metrics for measuring mosquito-borne pathogen transmission, and well-defined predictions about their quantitative relations;  
# - The longevity of adult mosquitoes; and
# - predictions about the responses and response timelines of various measures of transmission control.
# [[3]].
# 
# [3]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/
# 
# The most important inference of the above analysis, is that all the mosquitos must not be killed or all human population immunized in order to eradicate malaria. All we have to do is to disrupt breeding of the mosquitoes to the point that the basic reproduction rate of disease $R_0$ is very low. By so doing, the disease will then die out eventually, as the system tends to the "Disease-Free" state at the now-stabilized origin $(0,0)$.
# 
# By observing the dependence of $R_0 =\frac{a^2bcm}{r\mu_2}$ on the quantities $a,b,c,m,r$,and $\mu_2$, it will be beneficial to reduce **a, b, c** and $m$ (where $R_0$ is directly proportional to $a^2$), while **increasing** $\mu_2$ and $r$.
# 
# - Reduction in the longevity of mosquitoes ($\frac{1}{\mu_2}$),in other words increasing $\mu_2$, the death rate of mosquitoes as might be brought about by spraying, provide a sure if not the surest approach to reducing $R_0$, basic reproduction rate of the disease. Since an adult female mosquito must be alive in the first place in order to bite.
# - Without necessarily killing mosquitos, control strategies (screens, insecticides, and netting) that keep the vectors out of peoples' houses/beds, and thus reducing the biting rate of mosquitoes, reduce $R_0$. 
# - Drugs and vaccines that reduce human susceptibility reduce $b$; drugs that reduce the infectious period by killing plasmodia within the human body, promotes faster recovery, $r$; interupting the parasite's life cycle within mosquitoes, say by the creation of transgenic mosquitos, reduces $c$; reducing the overall mosquito population by spraying or draining marshes and other breeding sites as well as a reduction in the local human population density (say through relocation programs) reduces $m$ [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html. 
# 
# With the current sets of intervention, there will still be malaria prevalent in high transmission areas say Africa, even if the interventions were scaled up to a 100% level over a sustained period. When $R_0 < 1$, the transmission will tend towards 0 in the long run. 
# When $R_0 > 1$ then the malaria persists and spreads among the population, the proportions of infected humans and mosquitoes tend to their endemic levels [[2]].
# 
# [2]: https://www.slideshare.net/BenWielgosz/introduction-to-the-agroecology-of-malaria
# 
# Lack of resources and political instability can result in majority of a population being infected due to the lack of social programs to tackle malaria, there is also the problem of malaria parasites becoming increasingly resistant to antimalarial drugs [[6]].
# 
# [6]: https://www.slideshare.net/JordonTan/malaria-14058092?next_slideshow=1. 
# 
# In summary, **control strategies that focus** on the **death rate**, $\mu_2$, of adult mosquitoes or the **biting rate**, $a$, are **essentially more effective** than "non-$a$ or $\mu_2$" strategies. At the same time, "non-$a$ or $\mu_2$" control strategies must necessarily be more effective than $\mu_2$ or $a$ strategies to achieve the same results [[1]].
# 
# [1]: http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html.

# $$$$

# $$$$

# # ACKNOWLEDGEMENTS#

# The support, teachings and constructive comments of Dr. Michael P. Lamoureux of the Mathematics and Statistics Department, University of Calgary is highly appreciated and acknowledged. The authors would also like to thank Dr. Cristian Rios of the Mathematics and Statistics Department, University of Calgary for his extensive teachings on ODES (existence and uniqueness theorems, linearization of systems of equations and stability). Finally, the recommendation of the course by Dr. Ryan Hamilton, Instructor, Assistant Head - Undergraduate Mathematics, University of Calgary is without hesitation, very much appreciated as well.

# $$$$

# $$$$

# # REFERENCES#

# [1] The University of Arizona. 2006. *"Modelling Malaria."*. Biological Learning Center. Accessed April 11, 2019. Retrieved from http://www.blc.arizona.edu/courses/schaffer/182/Malaria/RossEqs.html.

# [2] Wielgosz, Benjamin. 2015. *"Introduction To The Agro-Ecology Of Malaria".* Slideshare.Net. Accessed April 11, 2019. Retrieved from https://www.slideshare.net/BenWielgosz/introduction-to-the-agroecology-of-malaria.

# [3] Smith David, Katherine Battle, Simon Hay, Christopher Barker, Thomas Scott, and Ellis Mckenzie. 2012. *"Ross, Macdonald, And A Theory For The Dynamics And Control Of Mosquito-Transmitted Pathogens".* Plos Pathogens. Accessed April 11, 2019. Retrieved from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3320609/.

# [4] Beckley Ross, Cametria Weatherspoon, Michael Alexander, Marissa Chandler, Anthony Johnson, and Ghan Bhatt. 2013. *"Modeling Epidemics With Differential Equations".* Ebook. Accessed April 11, 2019. Retrieved from http://www.tnstate.edu/mathematics/mathreu/filesreu/GroupProjectSIR.pdf.

# [5] Panfilov, Alexander V. *"Qualitative analysis of differential equations."* arXiv preprint arXiv:1803.05291 (2018).

# [6] Tan, Jordan. 2012. *"Malaria".* Slideshare.Net. Accessed April 11, 2019. Retrieved from https://www.slideshare.net/JordonTan/malaria-14058092?next_slideshow=1.
