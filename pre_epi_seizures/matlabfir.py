import matlab.engine as mleng

eng1 = mleng.start_matlab()

l=10
p=2
alpha=0.5

b = eng1.intfilt(l,p,alpha)

print b

eng1.quit()