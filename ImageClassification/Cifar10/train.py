import torch
import copy
import wandb
from tqdm import tqdm
from utils import dict_append


# =====================================
# Standard Training (Benchmark)
# =====================================

def train_step(epoch, net, trainloader, criterion, optimizer, device,opt_loss):
    '''
    Train single epoch.
    '''
    print(f'\nTraining epoch {epoch+1}..')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    convexity_gap = 0
    L = 0
    current_loss = 0
    prev_loss = 0
    num = 0
    denom = 0
    step = 0
    pbar = tqdm(enumerate(trainloader))
    # final_loss = 0
    # print(pbar)
    for batch, (inputs, labels) in pbar:
        # load data
        inputs, labels = inputs.to(device), labels.to(device)
        if step >0: #updating prev_prev_param
            prev_loss = current_loss
            optimizer.save_prev_param()
        # forward and backward propagation
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # if i ==0:
        #     prev_loss = loss.item()
        loss.backward()
        # if epoch ==0 and i == 0:
        #     optimizer.save_param()
        #     prev_loss = loss.item()
        # final_loss = loss.item()
        if step>0:
            linear_approx = optimizer.check_convexity()
            L = max(L, optimizer.check_smoothness())
            num += linear_approx
            denom +=loss.item() - prev_loss
            # ratio = linear_approx/(loss.item() - prev_loss)
            convexity_gap += loss.item() - prev_loss - linear_approx 
            # convexity_gap += ( current_convexity_gap - convexity_gap)/(i+1)
        optimizer.save_param()
        optimizer.step(closure = None)
        current_loss = loss.item()
        step+=1
        # stat updates
        # inner_product += inner
        train_loss += (loss.item() - train_loss)/(batch+1)  # average train loss
        total += labels.size(0)                             # total predictions
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()        # correct predictions
        train_acc = 100*correct/total                       # average train acc
        # prev_loss = loss.item()
        pbar.set_description(f'epoch {epoch+1} batch {batch+1}: \
            train loss {train_loss:.2f}, train acc: {train_acc:.2f}, smoothness_constant:{L:.2f}, convexity_gap:{convexity_gap:.2f} ' )
        # wandb.log({ 'smoothness_constant': L,'convexity_gap': convexity_gap })
    # optimizer_state = optimizer.state_dict()
    # optimizer.save_param()
    # return stats
    return train_loss, train_acc, convexity_gap/step, L, prev_loss, current_loss, num/denom


def test_step(epoch, net, testloader, criterion, device):
    '''
    Test single epoch.
    '''
    print(f'\nEvaluating epoch {epoch+1}..')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(testloader))
    with torch.no_grad():
        for batch, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += (loss.item() - test_loss)/(batch+1)
            total += labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            # test_acc = 100*correct/total
            # pbar.set_description(f'test loss: {test_loss}, test acc: {test_acc}')

    test_acc = 100*correct/total
    print(f'test loss: {test_loss}, test acc: {test_acc}')
    return test_loss, test_acc


def train(net, trainloader, testloader, epochs, criterion, optimizer, scheduler, device,args, opt_loss):
    stats = {
        'args': None,
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    inner_product = 0
    loss = float('inf')
    for epoch in range(epochs):
        train_loss, train_acc, convexity_gap, smoothness_constant, prev_loss, current_loss, ratio= train_step(epoch, net, trainloader, criterion, optimizer, device, opt_loss)
        # inner_product = inner
        if args.save:
            filename = "cifar10_resnet_checkpoint/" + str(epoch+1) + ".pth.tar"
            torch.save({'state_dict':optimizer.state_dict(), 'current_loss': current_loss, 'prev_loss': prev_loss
                            , 'model_dict': net.state_dict() }, filename)
            # save_param[epoch] = {'state_dict': copy.deepcopy(optimizer.state_dict()), 'loss':loss}
        test_loss, test_acc = test_step(epoch, net, testloader, criterion, device)
        scheduler.step()

        dict_append(stats, train_loss=train_loss, train_acc=train_acc, 
            test_loss=test_loss, test_acc=test_acc)
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc, 'convexity_gap': convexity_gap, 'smoothness_constant': smoothness_constant, 'innerProd/gap': ratio})
    return stats



# =====================================
# Private Training (DP-OTB)
# =====================================
'''
Train with DP-OTB.

To implement DP-OTB, we need to networks: one original network (net) corresponding to the OCO algorithm and
one clone network (net_clone) corresponding to DP-OTB.
*At the end, we copy net <- net_clone because we want to output the aggregated weights.

We stick to the notation in the paper, with k = 1 (i.e., beta_t = t).
'''

# def beta(t):
#     '''
#     \beta_t = t^k. Here we choose k = 1.
#     '''
#     return t


