from main import*

""" Testing """
#### Testing (model reload and show results)
save_path='D:/test_files/cycleGAN_datasets/monet2photo/monet2photo/records/'
loss_gen = loss_id = loss_gan = loss_cyc = 0.0
loss_disc = loss_disc_a = loss_disc_b = 0.0
tqdm_bar = tqdm(test_dataloader, desc=f'Testing Epoch {epoch} ', total=int(len(test_dataloader)))

for batch_idx, batch in enumerate(tqdm_bar):
    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))
    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

    ### Test Generators
    G_AB.eval()
    G_BA.eval()
    
    # Identity loss
    loss_id_A = criterion_identity(G_BA(real_A), real_A)
    loss_id_B = criterion_identity(G_AB(real_B), real_B)
    loss_identity = (loss_id_A + loss_id_B) / 2
    # GAN loss
    fake_B = G_AB(real_A)
    loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
    fake_A = G_BA(real_B)
    loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
    # Cycle loss
    recov_A = G_BA(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_AB(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)
    loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    # Total loss
    loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

    ### Test Discriminator-A
    D_A.eval()
    # Real loss
    loss_real = criterion_GAN(D_A(real_A), valid)
    # Fake loss (on batch of previously generated samples)
    fake_A_ = fake_A_buffer.push_and_pop(fake_A)
    loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
    # Total loss
    loss_D_A = (loss_real + loss_fake) / 2

    ### Test Discriminator-B
    D_B.eval()
    # Real loss
    loss_real = criterion_GAN(D_B(real_B), valid)
    # Fake loss (on batch of previously generated samples)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
    # Total loss
    loss_D_B = (loss_real + loss_fake) / 2
    loss_D = (loss_D_A + loss_D_B) / 2
    
    ### Log Progress
    loss_gen += loss_G.item(); 
    loss_id += loss_identity.item(); 
    loss_gan += loss_GAN.item(); 
    loss_cyc += loss_cycle.item()
    loss_disc += loss_D.item(); 
    loss_disc_a += loss_D_A.item(); 
    loss_disc_b += loss_D_B.item()
    tqdm_bar.set_postfix(Gen_loss=loss_gen/(batch_idx+1), identity=loss_id/(batch_idx+1), adv=loss_gan/(batch_idx+1), cycle=loss_cyc/(batch_idx+1),
                        Disc_loss=loss_disc/(batch_idx+1), disc_a=loss_disc_a/(batch_idx+1), disc_b=loss_disc_b/(batch_idx+1))
    
    # If at sample interval save image
    # if random.uniform(0,1)<1:
    if batch_idx % tqdm_bar==0:
        # Arrange images along x-axis
        real_A = make_grid(real_A, nrow=1, normalize=True)
        real_B = make_grid(real_B, nrow=1, normalize=True)
        fake_A = make_grid(fake_A, nrow=1, normalize=True)
        fake_B = make_grid(fake_B, nrow=1, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B), -1)
        save_image(image_grid, f"save_path{epoch}_{batch_idx}.png", normalize=False)

test_losses_gen.append(loss_gen/len(test_dataloader))
test_losses_disc.append(loss_disc/len(test_dataloader))

# Save model checkpoints
if np.argmin(test_losses_gen) == len(test_losses_gen)-1:
    # Save model checkpoints
    torch.save(G_AB.state_dict(), save_path+"G_AB.pth")
    torch.save(G_BA.state_dict(), save_path+"G_BA.pth")
    torch.save(D_A.state_dict(), save_path+"D_A.pth")
    torch.save(D_B.state_dict(), save_path+"D_B.pth")