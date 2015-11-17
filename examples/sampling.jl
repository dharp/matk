using PyCall
@pyimport matk

function dbexpl(p)
   t = [0:20:100]
   y = (p["par1"]*exp(-p["par2"]*t) + p["par3"]*exp(-p["par4"]*t))
end

p = matk.matk(model=dbexpl)
pycall(p[:add_par],PyAny,"par1",min=0,max=1)
pycall(p[:add_par],PyAny,"par2",min=0,max=0.2)
pycall(p[:add_par],PyAny,"par3",min=0,max=1)
pycall(p[:add_par],PyAny,"par4",min=0,max=0.2)

o = pycall(p[:forward],PyAny)

s = pycall(p[:lhs],PyAny,siz=500, seed=1000)



