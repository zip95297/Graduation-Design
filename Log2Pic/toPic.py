import matplotlib
import matplotlib.pyplot as plt
from Log2List import LogToDict
import random

class ToPic():

    def __init__(self, 
                 data:list,
                 isEpoch:bool=True,
                 EpochAvg:bool=True,
                 isLoss:bool=False, 
                 isHardLoss:bool=False, 
                 isSoftLoss:bool=False, 
                 isAcc:bool=False, 
                 isTh:bool=False,
                 x_step:int = 1, 
                 loss_color:str = "r", 
                 hard_loss_color:str = "g", 
                 soft_loss_color:str = "b", 
                 acc_color:str = "y", 
                 th_color:str = "c",
                 isGird:bool = False,
                 isRandomColor:bool = False
                 ):
        """
        基本描述：
            用于将数据绘制成图像,横坐标为batch，纵坐标为loss或者acc的大小
        详细描述：
            data: 从LogToDict中获取的数据
            isEpoch: batch维度还是epoch维度
            EpochAvg: 是否在epoch维度进行平均
            isLoss: 是否绘制loss
            isHardLoss: 是否绘制hard_loss
            isSoftLoss: 是否绘制soft_loss
            isAcc: 是否绘制acc
            isTh: 是否绘制th
            x_step: 横坐标的步长
            x_label: 横坐标的标签
            loss_color: loss的颜色
            hard_loss_color: hard_loss的颜色
            soft_loss_color: soft_loss的颜色
            acc_color: acc的颜色
            th_color: th的颜色
            isGrid: 是否绘制网格
            isRandomColor: 是否随机颜色
        """
        self.data = data
        self.total_train_batch = len(data)*25
        
        if isSoftLoss:
            for i in range(len(data)):
                self.data[i]['soft_loss'] = float(self.data[i]['soft_loss'])*15
        
        # each epoch have 284 instances
        if isEpoch:
            if EpochAvg :
            
                temp_data=[]

                epoch_num = int(data[-1]['epoch'])
                for i in range(1, epoch_num+1):

                    avg_loss = 0
                    avg_hard_loss = 0
                    avg_soft_loss = 0
                    avg_acc = 0
                    avg_th = 0
                    count = 0
                    for item in data :
                        if int(item['epoch']) != i :
                            continue
                        count+=1
                        if isLoss :
                            avg_loss += float(item['loss'])
                        if isHardLoss :
                            avg_hard_loss += float(item['hard_loss'])
                        if isSoftLoss :
                            avg_soft_loss += float(item['soft_loss'])
                        if isAcc :
                            avg_acc += float(item['accuracy'])
                        if isTh :
                            avg_th += float(item['threshold'])

                    avg_loss /= count
                    avg_hard_loss /= count
                    avg_soft_loss /= count
                    avg_acc /= count
                    avg_th /= count
                    temp_data.append({'model':data[0]['model'], 'epoch':i, 'loss':avg_loss, 'hard_loss':avg_hard_loss, 'soft_loss':avg_soft_loss, 'accuracy':avg_acc, 'threshold':avg_th})
                self.data = temp_data
            else :
                self.data = data[::284]
                
        else :
            for item in data :
                item['batch'] = int(item['batch']) + 7100 * (int(item['epoch'])-1)


        # 双引号之间使用小引号
        self.title = f"MODEL {data[0]['model']} ~{' LOSS' if isLoss else ''} {' HARD_LOSS' if isHardLoss else ''} {' SOFT_LOSS' if isSoftLoss else ''} {' ACC' if isAcc else ''} {' TH' if isTh else ''}"
        self.xlabel = "EPOCH" if isEpoch else "BATCH"
        self.ylabel = "VALUE"

        self.isEpoch = isEpoch
        self.isLoss = isLoss
        self.isHardLoss = isHardLoss
        self.isSoftLoss = isSoftLoss
        self.isAcc = isAcc
        self.isTh = isTh
        self.x_step = x_step
        if not isRandomColor:
            self.loss_color = loss_color
            self.hard_loss_color = hard_loss_color
            self.soft_loss_color = soft_loss_color
            self.acc_color = acc_color
            self.th_color = th_color
        else :
            self.loss_color = random_rgb()
            self.hard_loss_color = random_rgb()
            self.soft_loss_color = random_rgb()
            self.acc_color = random_rgb()
            self.th_color = random_rgb()

        self.linestyle = "-"  # 绘图的线条样式，可取值为：'-'（实线），'--'（虚线），':'（点线），'-.'（点划线）
        self.marker = ","  # 绘图的标记样式，可取值为：'.'（点），','（像素点），'o'（圆圈），'v'（倒三角），'^'（正三角），'<'（左三角），'>'（右三角），'1'（下箭头），'2'（上箭头），'3'（左箭头），'4'（右箭头），'s'（正方形），'p'（五角星），'*'（星号），'h'（六边形1），'H'（六边形2），'+'（加号），'x'（乘号），'D'（菱形），'d'（瘦菱形），'|'（竖线），'_'（横线）
        # self.legend = ""  # 绘图的图例，用于标识不同曲线的含义
        self.isGrid = isGird  # 绘图的网格显示

        self.x = [ d['epoch'] if isEpoch else d['batch'] for d in self.data]
        self.y_loss = [0]
        self.y_hard_loss = [0]
        self.y_soft_loss = [0]
        self.y_acc = [0]
        self.y_th = [0]


    def work_single(self):
        """
        处理数据并根据指定的参数创建绘图。

        该方法从`self.data`列表中提取所需的数据，并使用matplotlib创建绘图。
        绘图将具有指定的标题、x轴标签、y轴标签和x轴刻度标记。

        参数:
            无

        返回:
            无
        """
        if self.isLoss:
            self.y_loss = [float(data['loss']) for data in self.data]
        if self.isHardLoss:
            self.y_hard_loss = [float(data['hard_loss']) for data in self.data]
        if self.isSoftLoss:
            self.y_soft_loss = [float(data['soft_loss']) for data in self.data]
        if self.isAcc:
            self.y_acc = [float(data['accuracy']) for data in self.data]
        if self.isTh:
            self.y_th = [float(data['threshold']) for data in self.data]

        y_max = max(max(self.y_loss), max(self.y_hard_loss), max(self.y_soft_loss), max(self.y_acc), max(self.y_th))

        plt.figure(figsize=(10, 6))
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        if not self.isEpoch :
            plt.xticks(range(0, self.total_train_batch, 7115))
        # plt.yticks(range(0, int(y_max)+1, 1))
        # plt.grid(self.isGrid)

        if self.isLoss:
            plt.plot(self.x, self.y_loss, color=self.loss_color, linestyle=self.linestyle, marker=self.marker, label="LOSS")
        if self.isHardLoss:
            plt.plot(self.x, self.y_hard_loss, color=self.hard_loss_color, linestyle=self.linestyle, marker=self.marker, label="HARD_LOSS")
        if self.isSoftLoss: 
            plt.plot(self.x, self.y_soft_loss, color=self.soft_loss_color, linestyle=self.linestyle, marker=self.marker, label="SOFT_LOSS")
        if self.isAcc:
            plt.plot(self.x, self.y_acc, color=self.acc_color, linestyle=self.linestyle, marker=self.marker, label="ACC")
        if self.isTh:
            plt.plot(self.x, self.y_th, color=self.th_color, linestyle=self.linestyle, marker=self.marker, label="TH")

        plt.legend()
        plt.show()


def random_rgb():
    r = random.random()  # 随机生成红色分量，范围在 0 到 1 之间
    g = random.random()  # 随机生成绿色分量，范围在 0 到 1 之间
    b = random.random()  # 随机生成蓝色分量，范围在 0 到 1 之间
    return (r, g, b)        

def compare_model_train(data_list:list[LogToDict],additional_title:str=""):
    """
    基本描述：
        用于将多个模型的数据绘制成图像,横坐标维度一样，纵坐标任意
    详细描述：
        data_list: 从LogToDict中获取的数据
    """
    is_epoch_flag = data_list[0].isEpoch
    for self in data_list:
        if self.isEpoch != is_epoch_flag:
            print("ERROR: isEpoch should be the same")
            return
        
    plt.figure(figsize=(10, 6))
    title = f"MODEL COMPARE: "
    for self in data_list:
        title += f"{self.data[0]['model']} "
    title += additional_title
    plt.title(title)
    xlabel="EPOCH" if data_list[0].isEpoch else "BATCH"
    plt.xlabel(xlabel)
    plt.ylabel("VALUE")

    max_len=0
    for self in data_list:
        
        if self.isLoss:
            self.y_loss = [float(data['loss']) for data in self.data]
        if self.isHardLoss:
            self.y_hard_loss = [float(data['hard_loss']) for data in self.data]
        if self.isSoftLoss:
            self.y_soft_loss = [float(data['soft_loss']) for data in self.data]
        if self.isAcc:
            self.y_acc = [float(data['accuracy']) for data in self.data]
        if self.isTh:
            self.y_th = [float(data['threshold']) for data in self.data]

        y_max = max(max(self.y_loss), max(self.y_hard_loss), max(self.y_soft_loss), max(self.y_acc), max(self.y_th))

        if len(self.x) > max_len:
            if not self.isEpoch :
                plt.xticks(range(0, self.total_train_batch, 7115))
        max_len = max(max_len, len(self.x))
        # plt.yticks(range(0, int(y_max)+1, 1))
        # plt.grid(self.isGrid)

        if self.isLoss:
            plt.plot(self.x, self.y_loss, color=self.loss_color, linestyle=self.linestyle, marker=self.marker, label=f"{self.data[0]['model']} LOSS")
        if self.isHardLoss:
            plt.plot(self.x, self.y_hard_loss, color=self.hard_loss_color, linestyle=self.linestyle, marker=self.marker, label=f"{self.data[0]['model']} HARD_LOSS")
        if self.isSoftLoss: 
            plt.plot(self.x, self.y_soft_loss, color=self.soft_loss_color, linestyle=self.linestyle, marker=self.marker, label=f"{self.data[0]['model']} SOFT_LOSS")
        if self.isAcc:
            plt.plot(self.x, self.y_acc, color=self.acc_color, linestyle=self.linestyle, marker=self.marker, label=f"{self.data[0]['model']} ACC")
        if self.isTh:
            plt.plot(self.x, self.y_th, color=self.th_color, linestyle=self.linestyle, marker=self.marker, label=f"{self.data[0]['model']} TH")

    plt.legend()
    plt.show()

if __name__ == "__main__" :

    logToDict = LogToDict('./record_res18_arcface_alone')
    res, keys = logToDict.work()
    toPic1 = ToPic(res, isEpoch=False, EpochAvg=True, isAcc=True, isLoss=True, isTh=True, isRandomColor=True)
    #toPic.work_single()

    logToDict = LogToDict('./record_stu_with_T')
    res, keys = logToDict.work()
    toPic2 = ToPic(res, isEpoch=False, EpochAvg=False, isAcc=True, isLoss=True, isHardLoss=True, isSoftLoss=True, isTh=True, isRandomColor=True)
    #toPic.work_single()

    compare_model_train([toPic1, toPic2])

    plt.show()

    print("Done")