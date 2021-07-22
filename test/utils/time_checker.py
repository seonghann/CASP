import time

class DebugTimeCheck:
    def __init__(self,logger,function_name):
        self.logger = logger
        self.function_name = function_name
    
    def __call__(self,func):
        def decorator(*args,**kwargs):
            start_time = time.time()
            
            val = func(*args,**kwargs)
            done_time = time.time() - start_time
            msg = f'{self.function_name} is done in {done_time:.3f} secs\n' +\
                  f'args -> {args}\n' +\
                  f'kwargs -> {kwargs}'

            self.logger.debug(msg)
            return val
        return decorator
