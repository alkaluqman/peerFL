import numpy as np
import tensorflow as tf
from random import randint
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox, LabelOnlyGapAttack, LabelOnlyDecisionBoundary, ShadowModels
from art.utils import load_cifar10, to_categorical
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import os
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow_privacy
from sklearn.metrics import classification_report
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import BasicIterativeMethod
from art.defences.trainer import AdversarialTrainer

class SimpleMLP:

    @staticmethod
    def build(shape, classes, only_digits=True):
        base_model_1 = VGG16(include_top=False, input_shape=shape, classes=classes)
        model_1 = Sequential()
        model_1.add(base_model_1)  # Adds the base model (in this case vgg19 to model_1)
        model_1.add(
            Flatten())  # Since the output before the flatten layer is a matrix we have to use this function to get a
        # vector of the form nX1 to feed it into the fully connected layers
        # Add the Dense layers along with activation and TensorFlowV2Classifier
        # model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
        model_1.add(Dense(128, activation=('relu')))
        # model_1.add(Dropout(.2))
        model_1.add(Dense(10, activation=('softmax')))  # This is the classification layer
        return model_1

def local_training(client_num, local_model, build_flag, dp_flag, X_train, Y_train, X_test, Y_test):
    # Load client dataset from volume mounted folder
    # client_num = 1
    log_prefix = "[" + str(client_num).upper() + "] "
    x = 32
    y = 32
    z = 3
    input_shape = (x, y, z)
    num_classes = 10
    epoch=1
    batch_size=250
    epsilon,opt_order=0,0
    lr = 0.001
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    if dp_flag:
        l2_norm_clip = 1.5
        noise_multiplier = 0.447
        num_microbatches = 1
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=lr)
        epsilon,opt_order=compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=train_x.shape[0],
                            batch_size=batch_size,
                            noise_multiplier=noise_multiplier,
                            epochs=epoch,
                            delta=1e-5)
    else:
        optimizer = SGD(learning_rate=lr, momentum=0.9)
    if build_flag:
        print("%sBuilding model ..." % log_prefix)
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(shape=input_shape, classes=num_classes)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        local_model.build(input_shape=(None, x, y, z))

    print("%sTraining model ..." % log_prefix)
    # Training
    local_model.fit(X_train, Y_train, epochs=epoch, verbose=1, validation_data=(X_test, Y_test), batch_size=batch_size)
    print("%sDone" % log_prefix)

    # Save model - moved to node
    return local_model, epsilon, opt_order

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

def load_client_dataset():
    basepath = os.path.join(os.getcwd(), "p2pFLsim/Device/all_data")
    client_path = os.path.join(basepath, "saved_data_client_1")
    # client_path = "/usr/thisdocker/testset"
    print("[INFO] Loading from {} ".format(client_path))
    new_dataset = tf.data.experimental.load(client_path)
    return new_dataset

def rb_attack(art_classifier, X_train, Y_train, X_test, Y_test):
    print('Membership Inference Attack - Rule Based')
    rulebased_attack = MembershipInferenceBlackBoxRuleBased(art_classifier)

    inferred_train=rulebased_attack.infer(X_train, Y_train)
    inferred_test=rulebased_attack.infer(X_test, Y_test)

    # # # check accuracy
    train_acc = np.sum(inferred_train) / len(inferred_train)
    test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
    acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy: {test_acc:.4f}")
    print(f"Attack Accuracy: {acc:.4f}")
    # print rule-based precision and recall
    precision_recall=calc_precision_recall(np.concatenate((inferred_train, inferred_test)), 
                                np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test)))))
    print(f"Precision, Recall: {precision_recall}")

def bb_attack(art_classifier, X_train, Y_train, X_test, Y_test):
    print('Membership Inference Attack - Black Box')
    bb_attack = MembershipInferenceBlackBox(art_classifier)

    attack_train_ratio = 0.5
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_test_size = int(len(X_test) * attack_train_ratio)
    bb_attack.fit(X_train[:attack_train_size], Y_train[:attack_train_size],
                X_test[:attack_test_size], Y_test[:attack_test_size])
    
    inferred_train = bb_attack.infer(X_train[attack_train_size:], Y_train[attack_train_size:])
    inferred_test = bb_attack.infer(X_test[attack_test_size:], Y_test[attack_test_size:])

    # # # check accuracy
    train_acc = np.sum(inferred_train) / len(inferred_train)
    test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
    acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy: {test_acc:.4f}")
    print(f"Attack Accuracy: {acc:.4f}")
    # print rule-based precision and recall
    precision_recall=calc_precision_recall(np.concatenate((inferred_train, inferred_test)), 
                                np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test)))))
    print(f"Precision, Recall: {precision_recall}")

def gap_attack(art_classifier, X_train, Y_train, X_test, Y_test):
    print('Membership Inference Label-Only Attack - Gap')
    gapattack = LabelOnlyGapAttack(art_classifier)

    # infer attacked feature
    inferred_train = gapattack.infer(X_train, Y_train)
    inferred_test = gapattack.infer(X_test, Y_test)

    # check accuracy
    train_acc = np.sum(inferred_train) / len(inferred_train)
    test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
    acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy: {test_acc:.4f}")
    print(f"Attack Accuracy: {acc:.4f}")

    # print rule-based precision and recall
    precision_recall=calc_precision_recall(np.concatenate((inferred_train, inferred_test)), 
                                np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test)))))
    print(f"Precision, Recall: {precision_recall}")

def db_attack(art_classifier, X_train, Y_train, X_test, Y_test):
    print('Membership Inference Label-Only Attack - Decision Boundary')
    labelonly = LabelOnlyDecisionBoundary(art_classifier)

    print('Calibrating Distance Threshold')
    labelonly.calibrate_distance_threshold(X_train[:250], Y_train[:250],
                                            X_test[:250], Y_test[:250], max_iter=2, init_size=100)

    # get inferred values
    print("train")
    inferred_train_bb = labelonly.infer(X_train[49750:], Y_train[49750:], max_iter=1, init_size=50)
    print("test")
    inferred_test_bb = labelonly.infer(X_test[9500:], Y_test[9500:], max_iter=1, init_size=50)
    print('Calculate Performance Stats')
    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy {test_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")
    precision_recall=calc_precision_recall(np.concatenate((inferred_train_bb, inferred_test_bb)), 
                                np.concatenate((np.ones(len(inferred_train_bb)), np.zeros(len(inferred_test_bb)))))
    print(f"Precision, Recall: {precision_recall}")

def sm_attack(art_classifier, X_train, Y_train, X_test, Y_test):
    print('Membership Inference Shadow Models Attack')
    target_train_size = len(X_train) // 2
    x_target_train = X_train[:target_train_size]
    y_target_train = Y_train[:target_train_size]
    x_target_test = X_train[target_train_size:]
    y_target_test = Y_train[target_train_size:]
    print('Initializing Shadow Models')
    shadow_models = ShadowModels(art_classifier, num_shadow_models=3)

    print('Generating Shadow Dataset')
    X_test= np.reshape(X_test, (10000,-1,-1))
    #Y_test= np.reshape(Y_test, (32, 32,-1))
    print(X_test.shape)
    print(Y_test.shape)
    shadow_dataset = shadow_models.generate_shadow_dataset(X_test,to_categorical(Y_test))
    (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset

    # Shadow models' accuracy
    print([sm.model.score(x_target_test, y_target_test) for sm in shadow_models.get_shadow_models()])
    print('Fitting Membership Inference Attack on Shadow Models')
    attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
    attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)
    print('Calculating Performance Stats')
    member_infer = attack.infer(x_target_train, y_target_train)
    nonmember_infer = attack.infer(x_target_test, y_target_test)
    member_acc = np.sum(member_infer) / len(x_target_train)
    nonmember_acc = 1 - np.sum(nonmember_infer) / len(x_target_test)
    acc = (member_acc * len(x_target_train) + nonmember_acc * len(x_target_test)) / (len(x_target_train) + len(x_target_test))
    precision_recall = calc_precision_recall(np.concatenate((member_infer, nonmember_infer)), 
                                np.concatenate((np.ones(len(member_infer)), np.zeros(len(nonmember_infer)))))
    print('Attack Member Acc:', member_acc)
    print('Attack Non-Member Acc:', nonmember_acc)
    print('Attack Accuracy:', acc)
    print('Precision Recall: ',precision_recall)

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test), min_pixel_value, max_pixel_value = load_cifar10()
    train_len=50000
    test_len=10000
    malicious_nodes=[1,4,7]
    prev_model=None
    epsilonvals={}
    for attack in range(5):
        eps_list=[]
        for node in range(1,8):
            if node==7:
                t_u=train_len
                l_u=test_len
            else:
                t_u=int(train_len*(node/7))
                l_u=int(test_len*(node/7))
            t_l=int(train_len*(node/7)-train_len*(1/7))
            l_l=int(test_len*(node/7)-test_len*(1/7))
            train_x=X_train[t_l:t_u]
            train_y=Y_train[t_l:t_u]
            test_x=X_test[l_l:l_u]
            test_y=Y_test[l_l:l_u]
            if node==1:
                model,epsilon,opt_order=local_training(client_num=node, local_model=None, build_flag=True,dp_flag=True, X_train=train_x,Y_train=train_y,X_test=test_x,Y_test=test_y)
            else:
                model,epsilon,opt_order=local_training(client_num=node, local_model=prev_model, build_flag=False,dp_flag=True, X_train=train_x,Y_train=train_y,X_test=test_x,Y_test=test_y)
            prev_model=model
            if node in malicious_nodes:
                art_classifier = TensorFlowV2Classifier(model,input_shape=(32,32,3),nb_classes=10)
                if attack==0:
                    # Membership Inference Rule Based Attack
                    rb_attack(art_classifier, X_train, Y_train, X_test, Y_test)
                if attack==1:
                    # Membership Inference Black Box Attack
                    bb_attack(art_classifier, X_train, Y_train, X_test, Y_test)
                if attack==2:
                    # Membership Inference Label-Only Gap Attack
                    gap_attack(art_classifier, X_train, Y_train, X_test, Y_test)
                if attack==3:
                    # Membership Inference Label-Only Decision Boundary Attack
                    db_attack(art_classifier, X_train, Y_train, X_test, Y_test)
                if attack==4:
                    # Membership Inference Shadow Models Attack
                    sm_attack(art_classifier, X_train, Y_train, X_test, Y_test)
            else:
                print('Node ',node,' is an honest node')
                # predict probabilities for test set
            y_pred = model.predict(X_test, batch_size=64, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            Y_test_bool=np.argmax(Y_test, axis=1)
            eps_list.append(epsilon)
            print(classification_report(Y_test_bool, y_pred_bool))
        epsilonvals[attack]=eps_list  
    print(epsilonvals)