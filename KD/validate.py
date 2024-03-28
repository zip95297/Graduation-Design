import torch
from test import featurize, compute_accuracy
import torch.nn as nn

def validation_in_process(conf, student_model, teacher_model, criterion, groups, epoch):
    student_model.eval()
    
    with torch.no_grad():
        total_loss = 0
        total_samples = 0


        feature_dict_teacher = dict()
        feature_dict_student = dict()
        for group in groups:
            
            sd = featurize(group, conf.test_transform, student_model, conf.device)
            feature_dict_student.update(sd)
            td = featurize(group, conf.test_transform, teacher_model, conf.device)
            feature_dict_teacher.update(td)

            sorted_keys = sorted(feature_dict_teacher.keys())
    
            # val_teacher_outputs = feature_dict_teacher
            # val_student_outputs = feature_dict_student

            val_teacher_outputs = torch.stack([feature_dict_teacher[key] for key in sorted_keys])
            val_student_outputs = torch.stack([feature_dict_student[key] for key in sorted_keys])

            val_loss = criterion.validation_forward(val_student_outputs, val_teacher_outputs)
            total_loss += val_loss
            total_samples += len(group)
        validation_loss = total_loss / total_samples
        accuracy, threshold = compute_accuracy(feature_dict_student, conf.test_list, conf.test_root) 

        print(
            f"\t\tValidation Epoch: {epoch+1}/{conf.epoch}, "
            f"Validation Loss: {validation_loss:.8f}, "
            f"Accuracy: {accuracy:.5f}, "
            f"Threshold: {threshold:.5f}"
        )
        student_model.train()
        return accuracy, threshold
    
def validation_teacher_model(model, conf, groups):
    model.eval()
    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, conf.test_list, conf.test_root) 

    print(
        f"Test Teacher Model: {conf.teacher_model}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )