import logging

#def decorator_name1(function,):
#    def decorator_func(a,b):
#        print('before')
#        val = function(a,b)
#        print('after')
#        return val
#    return decorator_func
#
##my_function = decorator_name1(my_function)
#@decorator_name1
#def my_function1(a,b,):
#    print(a,b)
#    print('Operation -> plus')
#    return a+b

class Deco:
    def __init__(self,lg):
        self.lg = lg
    def __call__(self,func):

        def decorator(*args,**kwargs):
            self.lg.info('before')
            val = func(*args,**kwargs)
            self.lg.info('after')
            return val
        return decorator

class lg:

    def __init__(self,):
        print('AAA is generated')
    def info(self,msg):
        print(f'info : {msg}')

@Deco(lg())
def my_function2(a,b,):
    print(a,b)
    print('Operation -> plus')
    return a+b

val = my_function2(2,3)
print(val)

