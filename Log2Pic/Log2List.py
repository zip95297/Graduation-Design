

class LogToDict():
    """
    读取log文件，将信息保存到resDict中
    log文件要求：第0行保存了key的信息，之后每一行保存了对应的信息
    """
    def __init__(self,logFilePath):
        self.logFilePath = logFilePath
        self.key_list = []
        self.key_num = 0
        self.resDictList = []
        # resDict : [record_index][key:{model,epoch,batch,loss,accuracy,threshold,hard_loss,soft_loss,time}]

    def work(self) -> tuple[list ,list] : 
        """
        基本描述：
            读取log文件，将信息保存到resDict中
        详细描述：
            :param self: 类本身
            :type self: LogToDict
            :returns: resDictList, Keylist
            :rtype: list, list
        """
        self.resDictList = []
        with open(self.logFilePath, 'r') as f:
            lines = f.readlines()
            
            # 第0行保存了key的信息
            self.key_list=lines[0].split()
            self.key_num=len(self.key_list)
            # 读取完后删除，只保留信息
            lines.pop(0)

            for line in lines:
                instance = {}
                value = line.split(maxsplit=self.key_num-1)
                for i in range(self.key_num):
                    instance[self.key_list[i]] = value[i]
                self.resDictList.append(instance)
        return self.resDictList , self.key_list
    
if __name__ == "__main__" :
    logToDict = LogToDict('./record_res18_arcface_alone')
    resDictList = logToDict.work()

