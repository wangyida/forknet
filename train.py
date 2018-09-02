import numpy as np
import tensorflow as tf

from config import cfg
from util import DataProcess, scene_model_id_pair
from model import FCR_aGAN

def learning_rate(rate, step):
    if step < rate[1]:
        lr = rate[0]
    else:
        lr = rate[2]
    return lr


def train(n_epochs, learning_rate_G, learning_rate_D, batch_size, mid_flag, check_num):
    beta_G = cfg.TRAIN.ADAM_BETA_G
    beta_D = cfg.TRAIN.ADAM_BETA_D
    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0],n_vox[1],n_vox[2],dim[4]]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX 
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    freq = cfg.CHECK_FREQ
    record_vox_num = cfg.RECORD_VOX_NUM
    refine_ch = cfg.NET.REFINE_CH
    refine_kernel = cfg.NET.REFINE_KERNEL

    refine_start = cfg.SWITCHING_ITE

    fcr_agan_model = FCR_aGAN(
            batch_size=batch_size,
            vox_shape=vox_shape,
            dim_z=dim_z,
            dim=dim,
            start_vox_size=start_vox_size,
            kernel=kernel,
            stride=stride,
            refine_ch=refine_ch,
            refine_kernel = refine_kernel,
            )

    Z_tf, z_enc_tf, vox_tf, vox_gen_tf, vox_gen_decode_tf, vox_refine_dec_tf, vox_refine_gen_tf,\
     recons_loss_tf, code_encode_loss_tf, gen_loss_tf, discrim_loss_tf, recons_loss_refine_tf, gen_loss_refine_tf, discrim_loss_refine_tf,\
      cost_enc_tf, cost_code_tf, cost_gen_tf, cost_discrim_tf, cost_gen_ref_tf, cost_discrim_ref_tf, summary_tf,\
      tsdf_tf = fcr_agan_model.build_model()
    """
    z_enc_dep_tf, dep_tf, vox_gen_decode_dep_tf,\
    recons_dep_loss_tf, code_encode_dep_loss_tf, gen_dep_loss_tf, discrim_dep_loss_tf,\
    cost_enc_dep_tf, cost_code_dep_tf, cost_gen_dep_tf, cost_discrim_dep_tf, cost_code_compare_tf,\
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver(max_to_keep=cfg.SAVER_MAX)

    data_paths = scene_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION)
    print '---amount of data:' + str(len(data_paths))
    data_process = DataProcess(data_paths, batch_size, repeat = True)

    encode_vars = filter(lambda x: x.name.startswith('enc'), tf.trainable_variables())
    discrim_vars = filter(lambda x: x.name.startswith('discrim_vox'), tf.trainable_variables())
    # depth--start
    """
    depth_vars = filter(lambda x: x.name.startswith('dep'), tf.trainable_variables())
    discrim_dep_vars = filter(lambda x: x.name.startswith('discrim_dep'), tf.trainable_variables())
    """
    # depth--end
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    code_vars = filter(lambda x: x.name.startswith('cod'), tf.trainable_variables())
    refine_vars = filter(lambda x: x.name.startswith('refine'), tf.trainable_variables())

    lr_VAE=tf.placeholder(tf.float32, shape=[])
    train_op_encode=tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_D, beta2=0.9).minimize(
                    cost_enc_tf, var_list=encode_vars)
    train_op_discrim = tf.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                    cost_discrim_tf, var_list=discrim_vars, global_step=global_step)
    train_op_gen = tf.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                    cost_gen_tf, var_list=gen_vars)
    train_op_code = tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_G, beta2=0.9).minimize(
                    cost_code_tf, var_list=code_vars)
    # depth--start
    """
    train_op_latent_depvox = tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_G, beta2=0.9).minimize(
                    cost_code_compare_tf, var_list=depth_vars)
    train_op_encode_dep=tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_D, beta2=0.9).minimize(
                    cost_enc_dep_tf, var_list=depth_vars)
    train_op_discrim_dep = tf.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                    cost_discrim_dep_tf, var_list=discrim_dep_vars)
    train_op_gen_dep = tf.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                    cost_gen_dep_tf, var_list=gen_vars)
    train_op_code_dep = tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_G, beta2=0.9).minimize(
                    cost_code_dep_tf, var_list=code_vars)
    """
    # depth--end
    train_op_refine = tf.train.AdamOptimizer(
            lr_VAE, beta1=beta_G, beta2=0.9).minimize(
                    cost_gen_ref_tf, var_list=refine_vars)
    train_op_discrim_refine = tf.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                    cost_discrim_ref_tf, var_list=discrim_vars, global_step=global_step)

    Z_tf_sample, vox_tf_sample = fcr_agan_model.samples_generator(visual_size=batch_size)
    sample_vox_tf, sample_refine_vox_tf = fcr_agan_model.refine_generator(visual_size=batch_size)
    writer=tf.summary.FileWriter(cfg.DIR.LOG_PATH, sess.graph_def)
    tf.initialize_all_variables().run()

    if mid_flag:
        chckpt_path=cfg.DIR.CHECK_PT_PATH + str(check_num) #+ '-' + str(check_num * freq)
        saver.restore(sess, chckpt_path)
        Z_var_np_sample = np.load(cfg.DIR.TRAIN_OBJ_PATH + '/sample_z.npy').astype(np.float32)
        Z_var_np_sample = Z_var_np_sample[:batch_size]
        print '---weights restored'
    else:
        Z_var_np_sample = np.random.normal(size=(batch_size,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z)).astype(np.float32)
        np.save(cfg.DIR.TRAIN_OBJ_PATH + '/sample_z.npy', Z_var_np_sample)

    ite=check_num * freq +1
    cur_epochs=int(ite/int(len(data_paths)/batch_size))

    #training
    for epoch in np.arange(cur_epochs, n_epochs):
        epoch_flag=True
        while epoch_flag:
            print '=iteration:%d, epoch:%d' % (ite, epoch)
            db_inds, epoch_flag = data_process.get_next_minibatch()
            batch_voxel = data_process.get_voxel(db_inds)
            batch_voxel_train = batch_voxel
            batch_tsdf = data_process.get_tsdf(db_inds)
            batch_tsdf_train = np.expand_dims(batch_tsdf, axis=-1)
            """
            batch_depth = data_process.get_depth(db_inds)
            batch_depth_train = batch_depth / 255.0
            """
            lr=learning_rate(cfg.LEARNING_RATE_V, ite)

            batch_z_var = np.random.normal(size=(batch_size,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z)).astype(np.float32)

            if ite < refine_start:
                for s in np.arange(2):
                    _, recons_loss_val, code_encode_loss_val, cost_enc_val = sess.run(
                                [train_op_encode, recons_loss_tf, code_encode_loss_tf, cost_enc_tf],
                                feed_dict={vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train, Z_tf:batch_z_var, lr_VAE:lr},
                                )

                    _, gen_loss_val, cost_gen_val = sess.run(
                                [train_op_gen, gen_loss_tf, cost_gen_tf],
                                feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                                )
                    # depth--start
                    """
                    _, cost_code_compare_val = sess.run(
                            [train_op_latent_depvox, cost_code_compare_tf],
                            feed_dict={vox_tf:batch_voxel_train, dep_tf:batch_depth_train, lr_VAE:lr},
                            )
                    _, recons_dep_loss_val, code_encode_dep_loss_val, cost_enc_dep_val = sess.run(
                                [train_op_encode_dep, recons_dep_loss_tf, code_encode_dep_loss_tf, cost_enc_dep_tf],
                                feed_dict={vox_tf:batch_voxel_train, dep_tf:batch_depth_train, tsdf_tf:batch_tsdf_train, Z_tf:batch_z_var, lr_VAE:lr},
                                )
                    
                    _, gen_dep_loss_val, cost_gen_dep_val = sess.run(
                                [train_op_gen_dep, gen_dep_loss_tf, cost_gen_dep_tf],
                                feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, dep_tf:batch_depth_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                                )
                    # depth--end
                    """
                _, discrim_loss_val, cost_discrim_val = sess.run(
                            [train_op_discrim, discrim_loss_tf, cost_discrim_tf],
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train},
                            )

    	    
                _, cost_code_val, z_enc_val = sess.run(
                            [train_op_code, cost_code_tf, z_enc_tf],
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                            )
                """
                # depth--start
                _, discrim_dep_loss_val, cost_discrim_dep_val = sess.run(
                            [train_op_discrim_dep, discrim_dep_loss_tf, cost_discrim_dep_tf],
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, dep_tf:batch_depth_train, tsdf_tf:batch_tsdf_train},
                            )

    	    
                _, cost_code_dep_val, z_enc_dep_val= sess.run(
                            [train_op_code_dep, cost_code_dep_tf, z_enc_dep_tf],
                            feed_dict={Z_tf:batch_z_var, dep_tf:batch_depth_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                            )
                """
                # depth--end
                summary = sess.run(
                            summary_tf,
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                            )
    	    

                print 'reconstruction loss:', recons_loss_val if ('recons_loss_val' in locals()) else 'None'
                # print '            (depth):', recons_dep_loss_val if ('recons_dep_loss_val' in locals()) else 'None'

                print '   code encode loss:', code_encode_loss_val if ('code_encode_loss_val' in locals()) else 'None'

                # print '            (depth):', code_encode_dep_loss_val if ('code_encode_dep_loss_val' in locals()) else 'None'

                print '           gen loss:', gen_loss_val if ('gen_loss_val' in locals()) else 'None'

                # print '            (depth):', gen_dep_loss_val if ('gen_dep_loss_val' in locals()) else 'None'

                print '       cost_encoder:', cost_enc_val if ('cost_enc_val' in locals()) else 'None'

                # print '            (depth):', cost_enc_dep_val if ('cost_enc_dep_val' in locals()) else 'None'

                print '     cost_generator:', cost_gen_val if ('cost_gen_val' in locals()) else 'None'

                # print '            (depth):', cost_gen_dep_val if ('cost_gen_dep_val' in locals()) else 'None'

                print ' cost_discriminator:', cost_discrim_val if ('cost_discrim_val' in locals()) else 'None'

                # print '            (depth):', cost_discrim_dep_val if ('cost_discrim_dep_val' in locals()) else 'None'

                print '          cost_code:', cost_code_val if ('cost_code_val' in locals()) else 'None'

                # print '            (depth):', cost_code_dep_val if ('cost_code_dep_val' in locals()) else 'None'

                # print ' diff_codes_vox_dep:', cost_code_compare_val if ('cost_code_compare_val' in locals()) else 'None'

                print '   avarage of enc_z:', np.mean(np.mean(z_enc_val,4)) if ('z_enc_val' in locals()) else 'None'

                print '       std of enc_z:', np.mean(np.std(z_enc_val,4)) if ('z_enc_val' in locals()) else 'None'

                # print 'avarage of enc_z_dep:', np.mean(np.mean(z_enc_dep_val,4)) if ('z_enc_dep_val' in locals()) else 'None'

                # print '    std of enc_z_dep:', np.mean(np.std(z_enc_dep_val,4)) if ('z_enc_dep_val' in locals()) else 'None'


                if np.mod(ite, freq) == 0:
                    vox_models = sess.run(
                            vox_tf_sample,
                            feed_dict={Z_tf_sample:Z_var_np_sample},
                            )
                    vox_models_cat = np.argmax(vox_models, axis=4)
                    record_vox = vox_models_cat[:record_vox_num]
                    np.save(cfg.DIR.TRAIN_OBJ_PATH + '/' + str(ite/freq) + '.npy', record_vox)
                    save_path=saver.save(sess, cfg.DIR.CHECK_PT_PATH + str(ite/freq), global_step=None)

            else:
                _, recons_loss_val, recons_loss_refine_val, gen_loss_refine_val, cost_gen_ref_val = sess.run(
                            [train_op_refine, recons_loss_tf, recons_loss_refine_tf, gen_loss_refine_tf, cost_gen_ref_tf],
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train, lr_VAE:lr},
                            )

                _, discrim_loss_refine_val, cost_discrim_ref_val, summary = sess.run(
                            [train_op_discrim_refine, discrim_loss_refine_tf, cost_discrim_ref_tf, summary_tf],
                            feed_dict={Z_tf:batch_z_var, vox_tf:batch_voxel_train, tsdf_tf:batch_tsdf_train},
                            )

                print 'reconstruction loss:', recons_loss_val
                print ' recons refine loss:', recons_loss_refine_val
                print '           gen loss:', gen_loss_refine_val
                print ' cost_discriminator:', cost_discrim_ref_val

                if np.mod(ite, freq) == 0:
                    vox_models = sess.run(
                            vox_tf_sample,
                            feed_dict={Z_tf_sample:Z_var_np_sample},
                            )
                    refined_models = sess.run(
                            sample_refine_vox_tf,
                            feed_dict={sample_vox_tf:vox_models})
                    vox_models_cat = np.argmax(vox_models, axis=4)
                    record_vox = vox_models_cat[:record_vox_num]
                    np.save(cfg.DIR.TRAIN_OBJ_PATH + '/' + str(ite/freq) + '.npy', record_vox)

                    vox_models_cat = np.argmax(refined_models, axis=4)
                    record_vox = vox_models_cat[:record_vox_num]
                    np.save(cfg.DIR.TRAIN_OBJ_PATH + '/' + str(ite/freq) + '_refine.npy', record_vox)
                    save_path=saver.save(sess, cfg.DIR.CHECK_PT_PATH + str(ite/freq), global_step=None)


            
            writer.add_summary(summary, global_step=ite)



            ite +=1
