# This is a sample Python script.
from data.data_utils import *
from models.densenet import *
from utils import *
import time
import argparse

args = argparse.ArgumentParser("Chexpert GAN augmentation")
args.add_argument("--augmentation", choices=["no", "standard", "gan"], default="no")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opts = args.parse_args()
    if opts.augmentation == "no":
        train_dataloader, val_dataloader, LABELS = get_chexpert()
    elif opts.augmentation == "standard":
        train_dataloader, val_dataloader, LABELS = get_chexpert_standard()
    elif opts.augmentation == "gan":
        train_dataloader, val_dataloader, LABELS = get_chexpert_gan()
    else:
        print(f"No available augmentation name {opts.augmentation}")
        LABELS = None
        exit(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DenseNet121(num_classes=len(LABELS)).to(device)
    print(model)


    loss_criteria = nn.BCELoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Learning rate will be reduced automatically during training
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LEARNING_RATE_SCHEDULE_FACTOR,
                                                        patience=LEARNING_RATE_SCHEDULE_PATIENCE, mode='max',
                                                        verbose=True)

    # Best AUROC value during training
    best_score = 0
    model_path = "densenet.pth"
    training_losses = []
    validation_losses = []
    validation_score = []

    # Config progress bar
    mb = master_bar(range(MAX_EPOCHS))
    mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
    x = []

    nonimproved_epoch = 0
    start_time = time.time()

    # Training each epoch
    for epoch in mb:
        mb.first_bar.comment = f'Best AUROC score: {best_score}'
        x.append(epoch)

        # Training
        train_loss = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)
        mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
        training_losses.append(train_loss)

        # Evaluating
        val_loss, new_score = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)
        mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))
        validation_losses.append(val_loss)
        validation_score.append(new_score)

        # Update learning rate
        lr_scheduler.step(new_score)

        # Update training chart
        mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0, epoch + 1], [0, 1])

        # Save model
        if best_score < new_score:
            mb.write(f"Improve AUROC from {best_score} to {new_score}")
            best_score = new_score
            nonimproved_epoch = 0
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "epoch": epoch,
                        "lr_scheduler": lr_scheduler.state_dict()}, model_path)
        else:
            nonimproved_epoch += 1
        if nonimproved_epoch > 10:
            break
            print("Early stopping")
        if time.time() - start_time > 3600 * 8:
            break
            print("Out of time")

    # print(device)
    # while True:
    #     batch, label = next(data_loader)
    #     print(batch.size())
    #     print(label.size())
    #     break
    # print_hi('PyCharm')
    # data = pd.read_csv("/home/dzung/Downloads/archive/CheXpert-v1.0-small/train.csv")
    # # print(data.head())
    # train_data, val_data = train_test_split(data, test_size=0.1, random_state=2019)
    # # print(train_data)
    # # print(val_data)
    # train_dataset = ChestXrayDataset("../input/vietai-advanced-final-project-00/train/train", train_data, IMAGE_SIZE,
    #                                  True)
    #
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    #                               pin_memory=True)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
