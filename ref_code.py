
class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskE2E_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def init_net(self, args):
        # assert args.pretrained_path is not None
        if args.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A') \
                + tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_B')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverPart = tf.train.Saver(var, max_to_keep=5)

        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def encoder_spec(self, x, is_training=True, reuse=False, scope="part1"):
        channel = self.ngf
        feaMap_list, gamma_list, beta_list = [], [], []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                feaMap_list.append(x)
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)

            return feaMap_list, gamma_list, beta_list

    def ins_specific_branch(self, x, is_training=False, scope='VAEGAN_recon', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            feaMap_list, gamma_list, beta_list = self.encoder_spec(x, is_training, scope="part1")
            latent = self.share_encoder(feaMap_list[-1], is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            out = self.generator(out, is_training, scope="part4")
        return feaMap_list, gamma_list, beta_list, out, latent

    def _build_model(self):
        self._define_input()        
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A, specA, latent_specA = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B, specB, latent_specB = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory    
                fake_specA_logit = self.discriminator(self.fake_A_pool.query(specA), reuse=True, scope="discriminator_A")
                fake_specB_logit = self.discriminator(self.fake_A_pool.query(specB), reuse=True, scope="discriminator_B")
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")                
                fake_specA_logit = self.discriminator(specA, reuse=True, scope="discriminator_A")
                fake_specB_logit = self.discriminator(specB, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan) \
                    # + ops_UNIT.generator_loss(fake_specA_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan) \
                    # + ops_UNIT.generator_loss(fake_specB_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared) \
                    + ops_UNIT.KL_divergence(latent_specA) \
                    + ops_UNIT.KL_divergence(latent_specB)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) \
                    + ops_UNIT.L1_loss(specA, domain_A)
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) \
                    + ops_UNIT.L1_loss(specB, domain_B)
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 0e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_EncIN(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    ##############################################################################
    # BEGIN of ENCODERS
    def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
        channel = self.ngf
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn='instance', scope='resblock_'+str(i))

            return x
    # END of ENCODERS
    ##############################################################################

    def _build_model(self):
        self._define_input()        
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    # x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    # x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        # x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################
    
    def _build_model(self):
        self._define_input()        
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 0e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


## ref pip install guided-filter-tf
## https://github.com/wuhuikai/DeepGuidedFilter
from guided_filter_tf.guided_filter import fast_guided_filter
class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_GFhigh(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)

    def _build_model(self):
        self._define_input()
        domain_A_h4 = self.domain_A 
        domain_B_h4 = self.domain_B
        domain_A_h2 = tf.image.resize_nearest_neighbor(domain_A_h4, [int(0.5*self.img_h), int(0.5*self.img_w)])
        domain_B_h2 = tf.image.resize_nearest_neighbor(domain_B_h4, [int(0.5*self.img_h), int(0.5*self.img_w)])
        domain_A = tf.image.resize_nearest_neighbor(domain_A_h4, [int(0.25*self.img_h), int(0.25*self.img_w)])
        domain_B = tf.image.resize_nearest_neighbor(domain_B_h4, [int(0.25*self.img_h), int(0.25*self.img_w)])

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            self.x_ab_h2 = self._guided_filter(domain_A, x_ab, domain_A_h2, scope='guided_filter_A')
            self.x_ba_h2 = self._guided_filter(domain_B, x_ba, domain_B_h2, scope='guided_filter_B')
            self.x_ab_h4 = self._guided_filter(domain_A, x_ab, domain_A_h4, scope='guided_filter_A', reuse=True)
            self.x_ba_h4 = self._guided_filter(domain_B, x_ba, domain_B_h4, scope='guided_filter_B', reuse=True)

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 5e2*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, x_ab_h2, x_ba_h2, x_ab_h4, x_ba_h4, feaMapA, feaMapB = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.x_ab_h2, self.x_ba_h2, self.x_ab_h4, self.x_ba_h4, self.feaMask_list_A[0], self.feaMask_list_B[0]], \
            feed_dict={self.is_training : False}
        )

        real_A_img = unprocess_image(real_A, 127.5, 127.5)
        real_B_img = unprocess_image(real_B, 127.5, 127.5)
        fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
        fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
        fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
        fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
        fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
        fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
        x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
        x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
        x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
        x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
        x_ba_h2_img = unprocess_image(x_ba_h2, 127.5, 127.5)
        x_ab_h2_img = unprocess_image(x_ab_h2, 127.5, 127.5)
        x_ba_h4_img = unprocess_image(x_ba_h4, 127.5, 127.5)
        x_ab_h4_img = unprocess_image(x_ab_h4, 127.5, 127.5)
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A_img, [self.batch_size, 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B_img, [self.batch_size, 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_img, [self.batch_size, 1],
                    '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_img, [self.batch_size, 1],
                    '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_insNormed_img, [self.batch_size, 1],
                    '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_insNormed_img, [self.batch_size, 1],
                    '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa_img, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_img, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_img, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_img, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ba_h2_img, [self.batch_size, 1],
                    '{}/x_ba_h2_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_h2_img, [self.batch_size, 1],
                    '{}/x_ab_h2_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_h4_img, [self.batch_size, 1],
                    '{}/x_ba_h4_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_h4_img, [self.batch_size, 1],
                    '{}/x_ab_h4_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
## TODO: save mask ##
        b,h,w,c = feaMapA.shape
        # pdb.set_trace()
        for b_idx in range(b):
            mapA = np.tile(feaMapA[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            mapB = np.tile(feaMapB[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            save_images(mapA, [16, int(mapA.shape[0]/16)],
                        '{}/feaMapA0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_A.split(".")[0]))
            save_images(mapB, [16, int(mapB.shape[0]/16)],
                        '{}/feaMapB0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_B.split(".")[0]))

        for i in range(self.batch_size):
            # pdb.set_trace()
            fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
                [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
                feed_dict={self.is_training : False, 
                            self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
                            self.real_B : real_B}
            )
            fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
            fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
            x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
            x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
            save_images(fake_A_tmp_img, [self.batch_size, 1],
                        '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(fake_B_tmp_img, [self.batch_size, 1],
                        '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ba_tmp_img, [self.batch_size, 1],
                        '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ab_tmp_img, [self.batch_size, 1],
                        '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ShareSB(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def init_net(self, args):
        assert args.pretrained_path is not None
        if args.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverPart = tf.train.Saver(var, max_to_keep=5)

        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_ConvFeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def apply_feaMap_mask(self, x, feaMap, scope='conv_feaMask', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('UNIT', reuse=reuse):
            _, _, _, c = feaMap.get_shape().as_list()
            # feaMap = ops_UNIT.activation(x, activation_fn='leaky')
            feaMap = ops_UNIT.conv(feaMap, feaMap.get_shape().as_list()[-1], kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None)
            mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
            mask = mask/2. + 1./2. ## norm [0.5, 1] to [0.75, 1]
            mask = tf.image.resize_images(mask, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            x = x * mask
            return x

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

        counter = args.global_step
        start_time = time.time()

        if 0==args.global_step:
            if args.continue_train and self.load_last_ckpt(args.model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            ## global_step is set manually
            if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # batch_idxs = self.sample_num // self.batch_size
            batch_idxs = self.sample_num
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            # idx = int(epoch/len(self.weights_schedule))
            # loss_weights = self.weights_schedule[idx]

            d_loss = np.inf
            for idx in range(0, batch_idxs):
                step_ph = epoch*batch_idxs + idx
                num_steps = args.epoch*batch_idxs
                lr = args.lr*((1 - counter / num_steps)**0.9)
                # Update D
                # if d_loss>3.0:
                if self.D_optim is not None:
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
                                                            feed_dict={self.is_training : True, self.lr : lr})
                    self.writer.add_summary(summary_str, counter)

                # Update G
                for g_iter in range(args.G_update):
                # for g_iter in range(2):
                    fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], \
                                                                            feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_BlurFeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def apply_feaMap_mask(self, x, feaMap):
        # mask = (tf.sign(feaMap)+1.0)/2.0 ## select the >0 elements as mask
        mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
        # mask = mask/2. - 1 ## norm [0.5, 1] to [0, 1]
        mask = mask/2. + 1./2. ## norm [0.5, 1] to [0.75, 1]
        # mask = mask/5.*2. + 0.6 ## norm [0.5, 1] to [0.8, 1]
        # mask = mask/4. + 3./4. ## norm [0.5, 1] to [0.875, 1]
        # mask = tf.ones_like(mask) ## norm [0.5, 1] to [1, 1]
        mask = tf.image.resize_images(mask, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        mask = ops_UNIT.conv_gaussian_blur(mask, filter_size=3, sigma=3, name='gaussian_blur', padding='SAME')
        x = x * mask
        return x


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolShare(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    def share_encoder_generator(self, x, is_training=True, reuse=False, scope="share_encoder_generator"):
        channel = self.ngf * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            enc_list = []
            argmax_list = []
            # shape_list = [tf.shape(x)]
            for i in range(0, self.n_enc_share) :
                enc_list.append(x)
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='enc_resblock_'+str(i))
                if i < self.n_enc_share-1:
                    x, x_argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_'+str(i))
                    argmax_list.append(x_argmax)
                    # shape_list.append(tf.shape(x))

            x = ops_UNIT.gaussian_noise_layer(x)
            latent = x

            for i in range(0, self.n_gen_share) :
                if i < self.n_enc_share-1:
                    # pdb.set_trace()
                    # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                    x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='gen_resblock_'+str(i))


            return latent, x
    # END of SHARED LAYERS
    ##############################################################################

    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)
        shared, out = self.share_encoder_generator(out, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        shared, out = self.share_encoder_generator(out, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        shared, out = self.share_encoder_generator(out, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out, shared


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolEncGen(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    ##############################################################################
    # BEGIN of ENCODERS
    def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
        channel = self.ngf
        argmax_list = []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=1, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                x, x_argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_'+str(i))
                argmax_list.append(x_argmax)
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            return x, argmax_list
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, argmax_list, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)

        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        enc_A, argmax_list_A = self.encoder(x_A, self.is_training, scope="encoder_A")
        enc_B, argmax_list_B = self.encoder(x_B, self.is_training, scope="encoder_B")
        out = tf.concat([enc_A, enc_B], axis=0)
        argmax_list = [tf.concat([argmax_list_A[i], argmax_list_B[i]], axis=0) for i in range(len(argmax_list_A))]
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out, argmax_list = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, argmax_list, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out, argmax_list = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, argmax_list, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out, shared


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Pair(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(tf.concat([domain_B, domain_A], axis=-1), scope="discriminator_A")
            real_B_logit = self.discriminator(tf.concat([domain_A, domain_B], axis=-1), scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(tf.concat([domain_B, x_ba], axis=-1)), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(tf.concat([domain_A, x_ab], axis=-1)), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(tf.concat([domain_B, x_ba], axis=-1), reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(tf.concat([domain_A, x_ab], axis=-1), reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Triplet(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(tf.concat([real_A_logit,real_A_logit],axis=0), tf.concat([real_B_logit,fake_A_logit],axis=0), smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(tf.concat([real_B_logit,real_B_logit],axis=0), tf.concat([real_A_logit,fake_B_logit],axis=0), smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()




class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_LearnIN(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
    
    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=1:
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=1:
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ## TODO
    def learn_gamma_beta_feaMask(self, feaMap_list, gamma_list, beta_list, reuse=False, scope='leanrnIN'):
        with tf.variable_scope(scope, reuse=reuse) :
            return feaMap_list, gamma_list, beta_list

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            feaMap_list_A, gamma_list_A, beta_list_A = self.learn_gamma_beta_feaMask(self, feaMap_list_A, gamma_list_A, beta_list_A, scope='leanrnIN_A')
            feaMap_list_B, gamma_list_B, beta_list_B = self.learn_gamma_beta_feaMask(self, feaMap_list_B, gamma_list_B, beta_list_B, scope='leanrnIN_B')
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 0.*style_loss + 0.*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


## ref pip install guided-filter-tf
## https://github.com/wuhuikai/DeepGuidedFilter
from guided_filter_tf.guided_filter import fast_guided_filter
class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb
            # r=3
            # eps=1e-8
            x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A', reuse=True)
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B', reuse=True)
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm


        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, \
            x_aa, x_ba, x_ab, x_bb, feaMapA, feaMapB, x_aa_GF, x_ba_GF, x_ab_GF, x_bb_GF = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.feaMask_list_A[0], self.feaMask_list_B[0], \
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF], \
            feed_dict={self.is_training : False}
        )

        real_A_img = unprocess_image(real_A, 127.5, 127.5)
        real_B_img = unprocess_image(real_B, 127.5, 127.5)
        fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
        fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
        fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
        fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
        fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
        fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
        x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
        x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
        x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
        x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
        x_aa_GF_img = unprocess_image(x_aa_GF, 127.5, 127.5)
        x_ba_GF_img = unprocess_image(x_ba_GF, 127.5, 127.5)
        x_ab_GF_img = unprocess_image(x_ab_GF, 127.5, 127.5)
        x_bb_GF_img = unprocess_image(x_bb_GF, 127.5, 127.5)
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A_img, [self.batch_size, 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B_img, [self.batch_size, 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_img, [self.batch_size, 1],
                    '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_img, [self.batch_size, 1],
                    '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_insNormed_img, [self.batch_size, 1],
                    '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_insNormed_img, [self.batch_size, 1],
                    '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa_img, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_img, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_img, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_img, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa_GF_img, [self.batch_size, 1],
                    '{}/x_aa_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_GF_img, [self.batch_size, 1],
                    '{}/x_ba_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_GF_img, [self.batch_size, 1],
                    '{}/x_ab_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_GF_img, [self.batch_size, 1],
                    '{}/x_bb_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
## TODO: save mask ##
        b,h,w,c = feaMapA.shape
        # pdb.set_trace()
        for b_idx in range(b):
            mapA = np.tile(feaMapA[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            mapB = np.tile(feaMapB[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            save_images(mapA, [16, int(mapA.shape[0]/16)],
                        '{}/feaMapA0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_A.split(".")[0]))
            save_images(mapB, [16, int(mapB.shape[0]/16)],
                        '{}/feaMapB0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_B.split(".")[0]))

        for i in range(self.batch_size):
            # pdb.set_trace()
            fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
                [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
                feed_dict={self.is_training : False, 
                            self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
                            self.real_B : real_B}
            )
            fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
            fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
            x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
            x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
            save_images(fake_A_tmp_img, [self.batch_size, 1],
                        '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(fake_B_tmp_img, [self.batch_size, 1],
                        '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ba_tmp_img, [self.batch_size, 1],
                        '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ab_tmp_img, [self.batch_size, 1],
                        '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            

class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ConvGF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF):
    def _guided_map(self, x, is_training, activation_fn=ops_UNIT.LeakyReLU):
        channel_num = self.ngf
        x = slim.conv2d(x, channel_num, 1, 1, activation_fn=None)
        x = ops_UNIT.adaptive_BN(x, is_training)
        # x = ops_UNIT.adaptive_IN(x)
        x = activation_fn(x)
        x = slim.conv2d(x, 3, 1, 1, activation_fn=None)
        return x

    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        with tf.variable_scope(scope, reuse=reuse) :
            x_in = tf.concat([lr_x, hr_x], axis=0)
            x_out = self._guided_map(x_in, self.is_training)
            lr_x, hr_x = tf.split(x_out, 2, axis=0)
            return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF):
    def init_net(self, args):
        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        if args.pretrained_common_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverCommon = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def _dynamic_filters(self, x, is_training, filter_size=9, activation_fn=ops_UNIT.LeakyReLU, use_dynamic_bias=False):
        channel_num = self.ngf
        x = ops_UNIT.conv(x, channel_num, kernel=3, stride=1, pad=1, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_in')
        for i in range(0, 3) :
            x = ops_UNIT.resblock(x, channel_num, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                         normal_weight_init=self.normal_weight_init,
                         is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
        # dynamic_filters = ops_UNIT.conv(x, filter_size**2, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_filter')
        # dynamic_filters = tf.nn.softmax(dynamic_filters, axis=-1)

        kernel_1 = tf.Variable(tf.zeros([1, 1, channel_num, filter_size**2]))
        conv_kernel_1 = tf.nn.conv2d(x, kernel_1, [1,1,1,1], padding='SAME')
        dense = tf.sparse_to_dense(sparse_indices=[int((filter_size**2)/2)], output_shape=[filter_size**2], sparse_values=[1.0])
        biases_1 = tf.Variable(dense)
        dynamic_filters = tf.nn.bias_add(conv_kernel_1, biases_1)

        if use_dynamic_bias:
            dynamic_bias = ops_UNIT.conv(x, 1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_bias')
        else:
            dynamic_bias = 0

        return dynamic_filters, dynamic_bias

    def _guided_filter(self, x, G, hr_x=None, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        with tf.variable_scope(scope, reuse=reuse) :
            filter_size = 5
            pad = int((filter_size-1)/2)
            x_out = tf.zeros_like(G)
            B, H, W, C = x_out.get_shape().as_list()
            dynamic_filters, dynamic_bias = self._dynamic_filters(x, self.is_training, filter_size)
            G = tf.pad(G+dynamic_bias, [[0,0], [pad, pad], [pad, pad], [0,0]], "REFLECT")
            # convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding='VALID')
            # patches = tf.reshape(tf.extract_image_patches(x, [1, filter_size, filter_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"), [B,H,W,C,-1])
            # filter_patches = tf.tile(tf.expand_dims(dynamic_filters, axis=-2), [1,1,1,C,1])
            # x_out = tf.reduce_sum(filter_patches*patches, axis=[-1], keepdims=False)
            patches = tf.reshape(tf.extract_image_patches(G, [1, filter_size, filter_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"), [B,H,W,-1,C])
            filter_patches = tf.tile(tf.expand_dims(dynamic_filters, axis=-1), [1,1,1,1,C])
            x_out = tf.reduce_sum(filter_patches*patches, axis=[-2], keepdims=False)
            return x_out

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('GuidedFilter'):
            # r=3
            # eps=1e-8
            x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A', reuse=True)
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B', reuse=True)
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('UNIT'):
            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm


        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        # G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        G_vars = [var for var in t_vars if ('GuidedFilter' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_common_path is not None:
            self.saverCommon.restore(self.sess, args.pretrained_common_path)
            print('restored from pretrained_common_path:', args.pretrained_common_path)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

        counter = args.global_step
        start_time = time.time()


        if 0==args.global_step:
            if args.continue_train and self.load_last_ckpt(args.model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            ## global_step is set manually
            if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # batch_idxs = self.sample_num // self.batch_size
            batch_idxs = self.sample_num
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            # idx = int(epoch/len(self.weights_schedule))
            # loss_weights = self.weights_schedule[idx]

            d_loss = np.inf
            for idx in range(0, batch_idxs):
                step_ph = epoch*batch_idxs + idx
                num_steps = args.epoch*batch_idxs
                lr = args.lr*((1 - counter / num_steps)**0.9)
                # Update D
                # if d_loss>3.0:
                if self.D_optim is not None:
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
                                                            feed_dict={self.is_training : True, self.lr : lr})
                    self.writer.add_summary(summary_str, counter)

                # Update G
                for g_iter in range(args.G_update):
                    fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], \
                                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF_OnlyCrossDomain(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF):

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('GuidedFilter'):
            # r=3
            # eps=1e-8
            # x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A')
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            # x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B')
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('UNIT'):
            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        # G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        G_vars = [var for var in t_vars if ('GuidedFilter' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()



class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_Grad(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def gradient_map(self, x):
        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        _, _, _, ch = x.get_shape().as_list()
        kernels = []
        kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), ch) )
        kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), ch) )
        kernel = tf.constant(np.stack(kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        gradient_map = tf.abs(tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding="VALID"))
        # pdb.set_trace()
        return gradient_map

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm
            self.grad_A_map = self.gradient_map(domain_A)
            self.grad_B_map = self.gradient_map(domain_B)
            self.grad_ab_map = self.gradient_map(self.x_ab)
            self.grad_ba_map = self.gradient_map(self.x_ba)

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        grad_loss_a = tf.reduce_mean(ops_UNIT.L2_loss(self.grad_ab_map, self.grad_A_map))
        grad_loss_b = tf.reduce_mean(ops_UNIT.L2_loss(self.grad_ba_map, self.grad_B_map))
        self.grad_loss = grad_loss_a + grad_loss_b

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 500.0*self.grad_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.grad_loss_a_sum = tf.summary.scalar("grad_loss_a", grad_loss_a)
        self.grad_loss_b_sum = tf.summary.scalar("grad_loss_b", grad_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.grad_loss_a_sum, self.grad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, grad_A_map, grad_B_map, grad_ab_map, grad_ba_map = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.grad_A_map, self.grad_B_map, self.grad_ab_map, self.grad_ba_map], \
            feed_dict={self.is_training : False}
        )
        real_A_img = unprocess_image(real_A, 127.5, 127.5)
        real_B_img = unprocess_image(real_B, 127.5, 127.5)
        fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
        fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
        fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
        fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
        fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
        fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
        x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
        x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
        x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
        x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
        grad_A_map_img = unprocess_image((grad_A_map/3.-0.5)*2, 127.5, 127.5) ## gradient value is in about [0,3], norm to [-1,1]
        grad_B_map_img = unprocess_image((grad_B_map/3.-0.5)*2, 127.5, 127.5)
        grad_ab_map_img = unprocess_image((grad_ab_map/3.-0.5)*2, 127.5, 127.5)
        grad_ba_map_img = unprocess_image((grad_ba_map/3.-0.5)*2, 127.5, 127.5)
        grad_A_map_img = np.tile(np.mean(grad_A_map_img, keepdims=True), [1,1,1,3])
        grad_B_map_img = np.tile(np.mean(grad_B_map_img, keepdims=True), [1,1,1,3])
        grad_ab_map_img = np.tile(np.mean(grad_ab_map_img, keepdims=True), [1,1,1,3])
        grad_ba_map_img = np.tile(np.mean(grad_ba_map_img, keepdims=True), [1,1,1,3])
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A_img, [self.batch_size, 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B_img, [self.batch_size, 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_img, [self.batch_size, 1],
                    '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_img, [self.batch_size, 1],
                    '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_insNormed_img, [self.batch_size, 1],
                    '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_insNormed_img, [self.batch_size, 1],
                    '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa_img, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_img, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_img, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_img, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        try:
            # pdb.set_trace()
            save_images(grad_A_map_img, [self.batch_size, 1],
                        '{}/grad_A_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_B_map_img, [self.batch_size, 1],
                        '{}/grad_B_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_ab_map_img, [self.batch_size, 1],
                        '{}/grad_ab_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_ba_map_img, [self.batch_size, 1],
                        '{}/grad_ba_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

        for i in range(self.batch_size):
            # pdb.set_trace()
            fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
                [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
                feed_dict={self.is_training : False, 
                            self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
                            self.real_B : real_B}
            )
            fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
            fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
            x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
            x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
            save_images(fake_A_tmp_img, [self.batch_size, 1],
                        '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(fake_B_tmp_img, [self.batch_size, 1],
                        '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ba_tmp_img, [self.batch_size, 1],
                        '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ab_tmp_img, [self.batch_size, 1],
                        '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def init_net(self, args):
        # assert args.pretrained_vgg_path is not None
        # if args.pretrained_vgg_path is not None:
        #     var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
        #     var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
        #     self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    # def vgg_net(self, x, is_training=False, reuse=False):
    #     _, end_points = nets.vgg.vgg_19_style_full(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
    #     # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
    #     # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
    #     # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
    #     # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
    #     layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
    #     # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
    #     fea_list, gamma_list, beta_list = [], [], []
    #     # for i in [1, 2, 3]:
    #     # for i in [1, 2, 3, 4]:
    #     for i in [1, 2, 3, 4, 5]:
    #     # for i in [3]: 
    #         fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
    #         mean, var = tf.nn.moments(fea, [1,2])
    #         gamma = mean - 1.
    #         beta = var
    #         fea_list.append(fea)
    #         gamma_list.append(gamma)
    #         beta_list.append(beta)
        
    #     return fea_list, gamma_list, beta_list

    def vgg_net_tf_official(self, x, is_training=False, reuse=False):
        assert self.pretrained_vgg_path is not None
        def vgg_preprocess(x): ## from [-1,1] back to [0,255], then subtracts vgg mean
            VGG_MEAN = [123.68, 116.779, 103.939]     # This is R-G-B for Imagenet
            x = unprocess_image(x, 127.5, 127.5)
            x_R, x_G, x_B = tf.split(x, 3, axis=-1)
            x_R -= VGG_MEAN[0]
            x_G -= VGG_MEAN[1]
            x_B -= VGG_MEAN[2]
            x = tf.concat([x_R, x_G, x_B], axis=-1)
            # x = tf.concat([x_B, x_G, x_R], axis=-1)
            return x

        x = vgg_preprocess(x) 
        _, end_points = nets.vgg.vgg_19_style_full(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
        layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
        # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
        fea_list, gamma_list, beta_list = [], [], []
        # for i in [1, 2, 3]:
        # for i in [1, 2, 3, 4]:
        for i in [1, 2, 3, 4, 5]:
        # for i in [3]: 
            fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
            mean, var = tf.nn.moments(fea, [1,2])
            gamma = mean - 1.
            beta = var
            fea_list.append(fea)
            gamma_list.append(gamma)
            beta_list.append(beta)
        
        return fea_list, gamma_list, beta_list

    def vgg_net(self, x, is_training=False, reuse=False, scope="style_content_vgg"):
        vgg = tensorflow_vgg19.Vgg19_style('./weights/vgg19.npy')
        with tf.variable_scope(scope, reuse=reuse):
            ## x should be rgb image [batch, height, width, 3] values scaled [0, 1]
            x = (x+1.0)/2.0 ## preprocess
            vgg.build(x)
            fea_list = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
            ## L2 norm by channel
            try:
                fea_list = [tf.nn.l2_normalize(fea, axis=[1,2])for fea in fea_list]
            except:
                fea_list = [tf.nn.l2_normalize(fea, dim=[1,2])for fea in fea_list]


            # fea_list[0] = tf.Print(fea_list[0], [tf.reduce_max(fea_list[0]),tf.reduce_min(fea_list[0])], summarize=40, message="fea_list[0] is:")
            # pdb.set_trace()
            gamma_list, beta_list = [], [] 
            for fea in fea_list:
                mean, var = tf.nn.moments(fea, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)
        return fea_list, gamma_list, beta_list

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE):
        x = tf.concat([x_A, x_B, x_ab, x_ba], axis=0)
        feaMap_list, gamma_list, beta_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list_A = [tf.split(feaMap, 4, axis=0)[0] for feaMap in feaMap_list]
        feaMap_list_B = [tf.split(feaMap, 4, axis=0)[1] for feaMap in feaMap_list]
        feaMap_list_ab = [tf.split(feaMap, 4, axis=0)[2] for feaMap in feaMap_list]
        feaMap_list_ba = [tf.split(feaMap, 4, axis=0)[3] for feaMap in feaMap_list]

        gamma_list_A = [tf.split(gamma, 4, axis=0)[0] for gamma in gamma_list]
        gamma_list_B = [tf.split(gamma, 4, axis=0)[1] for gamma in gamma_list]
        gamma_list_ab = [tf.split(gamma, 4, axis=0)[2] for gamma in gamma_list]
        gamma_list_ba = [tf.split(gamma, 4, axis=0)[3] for gamma in gamma_list]
        
        beta_list_A = [tf.split(beta, 4, axis=0)[0] for beta in beta_list]
        beta_list_B = [tf.split(beta, 4, axis=0)[1] for beta in beta_list]
        beta_list_ab = [tf.split(beta, 4, axis=0)[2] for beta in beta_list]
        beta_list_ba = [tf.split(beta, 4, axis=0)[3] for beta in beta_list]

        num = len(feaMap_list_A)
        ## Use source domain ecnoder to extract the content feature map
        content_loss = 0.
        for i in range(num):
            content_loss += tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_A[i], feaMap_list_ab[i])) \
                        + tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_B[i], feaMap_list_ba[i]))

        ## Use target domain ecnoder to extract the style stastics, i.e. mean and var
        style_loss = 0.
        for i in range(num):
            style_loss += ops_UNIT.L2_loss(gamma_list_ab[i], gamma_list_B[i]) \
                        + ops_UNIT.L2_loss(beta_list_ab[i], beta_list_B[i])
            style_loss += ops_UNIT.L2_loss(gamma_list_ba[i], gamma_list_A[i]) \
                        + ops_UNIT.L2_loss(beta_list_ba[i], beta_list_A[i])

        # content_loss = tf.Print(content_loss, [content_loss], summarize=40, message="content_loss is:")
        # style_loss = tf.Print(style_loss, [style_loss], summarize=40, message="style_loss is:")

        return content_loss, style_loss

    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel_original = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            # channel = self.channel_list[-1]
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    channel = self.channel_list[-1-i]
                    if x.get_shape().as_list()[-1]!= channel:
                        x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)
                elif channel_original!=x.get_shape().as_list()[-1]:
                    channel = channel_original
                    x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                      
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
        channel_original = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            # channel = self.channel_list[-1]
            # x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans')
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    channel = self.channel_list[-1-i]
                    if x.get_shape().as_list()[-1]!= channel:
                        x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                    
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)
                elif channel_original!=x.get_shape().as_list()[-1]:
                    channel = channel_original
                    x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                                    
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################
    
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.vgg_net(domain_A, is_training=False, reuse=False)
        feaMap_list_B, gamma_list_B, beta_list_B = self.vgg_net(domain_B, is_training=False, reuse=True)
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        self.channel_list = [feaMap_list_A[i].get_shape().as_list()[-1] for i in range(len(feaMap_list_A))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e7*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        # self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        # self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, ])
                        # self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()



class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskFromOther_EncStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
    def translation(self, x_A, x_B, feaMap_list_A, feaMap_list_A_other, gamma_list_A, beta_list_A, feaMap_list_B, feaMap_list_B_other, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B_other, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A_other, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        feaMap_list_A_other, _, _ = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        feaMap_list_B_other, _, _ = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, feaMap_list_A_other, gamma_list_A, beta_list_A, feaMap_list_B, feaMap_list_B_other, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1.*style_loss + 0.2*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()



class UNIT_VAEGAN_LatentTransform(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def Latent_Transform(self, x, is_training=True, reuse=False, scope="Latent_Transform", activation_fn=ops_UNIT.LeakyReLU):
        _, out_h, out_w, out_c = x.get_shape().as_list()

        with tf.variable_scope(scope, reuse=reuse) :
            channel_num = out_c
            channel_step = self.ngf
            _, h, w, _ = x.get_shape().as_list()
            repeat_num = int(np.log2(h/4))

            # Encoder
            encoder_layer_list = []
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)

            for idx in range(repeat_num):
                res = x
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                encoder_layer_list.append(x)
                if idx < repeat_num - 1:
                    # channel_num *= 2
                    channel_num += channel_step
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn)

            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [x_shape[0], np.prod(x_shape[1:])])
            z = x = slim.fully_connected(x, 128, activation_fn=None)

            ## Decoder
            x = slim.fully_connected(z, x_shape[1]*x_shape[2]*channel_num, activation_fn=None)
            x = tf.reshape(x, x_shape)
            
            for idx in range(repeat_num):
                # pdb.set_trace()
                x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
                res = x
                # channel_num = hidden_num * (repeat_num-idx)
                channel_num = x.get_shape().as_list()[-1]
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                if idx < repeat_num - 1:
                    _, h, w, _ = x.get_shape().as_list()
                    x = tf.image.resize_nearest_neighbor(x, [2*h, 2*w])
                    # channel_num -= channel_step
                    channel_num /= 2
                    x = slim.conv2d(x, channel_num, 1, 1, activation_fn=activation_fn)

            out = slim.conv2d(x, out_c, 3, 1, activation_fn=None)

            return out

    def VAEGAN_recon(self, x, latent=None, is_training=True, scope='VAEGAN_recon', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if latent is None:
                enc = self.encoder(x, is_training, scope="part1")
                latent = self.share_encoder(enc, is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            out = self.generator(out, is_training, scope="part4")

        return out, latent

    def _build_model(self):
        self._define_input()
        # self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        self.x_aa, latent_A = self.VAEGAN_recon(domain_A, is_training=False, scope='VAEGAN_recon_A')
        self.x_bb, latent_B = self.VAEGAN_recon(domain_B, is_training=False, scope='VAEGAN_recon_B')
        # pdb.set_trace() 
        latent_ab = self.Latent_Transform(latent_A, self.is_training, scope="Latent_Transform_A2B")
        latent_ba = self.Latent_Transform(latent_B, self.is_training, scope="Latent_Transform_B2A")
        self.x_ab, _ = self.VAEGAN_recon(None, latent_ab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.x_ba, _ = self.VAEGAN_recon(None, latent_ba, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        self.fake_B, _ = self.VAEGAN_recon(None, latent_A, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.fake_A, _ = self.VAEGAN_recon(None, latent_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)


        _, latent_m = self.VAEGAN_recon(self.x_ab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        _, latent_n = self.VAEGAN_recon(self.x_ba, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        # pdb.set_trace() 
        latent_bab = self.Latent_Transform(latent_n, self.is_training, scope="Latent_Transform_A2B", reuse=True)
        latent_aba = self.Latent_Transform(latent_m, self.is_training, scope="Latent_Transform_B2A", reuse=True)
        self.x_bab, _ = self.VAEGAN_recon(None, latent_bab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.x_aba, _ = self.VAEGAN_recon(None, latent_aba, is_training=False, scope='VAEGAN_recon_A', reuse=True)

        real_latent_A_logit = self.discriminator(latent_A, scope="discriminator_latent_A")
        real_latent_B_logit = self.discriminator(latent_B, scope="discriminator_latent_B")
        fake_latent_A_logit = self.discriminator(latent_ba, reuse=True, scope="discriminator_latent_A")
        fake_latent_B_logit = self.discriminator(latent_ab, reuse=True, scope="discriminator_latent_B")

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_latent_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_latent_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_latent_A_logit, fake_latent_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_latent_B_logit, fake_latent_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss_a = ops_UNIT.KL_divergence(latent_A)
        # enc_loss_b = ops_UNIT.KL_divergence(latent_B)

        # l1_loss_a = ops_UNIT.L1_loss(self.x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(self.x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(self.x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(self.x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba
                           # self.L1_weight * l1_loss_a + \
                           # self.KL_weight * enc_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab
                           # self.L1_weight * l1_loss_b + \
                           # self.KL_weight * enc_loss_b

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'VAEGAN_recon' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        # self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        # self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        # self.enc_loss_a_sum = tf.summary.scalar("KL_enc_loss_a", enc_loss_a)
        # self.enc_loss_b_sum = tf.summary.scalar("KL_enc_loss_b", enc_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()




class UNIT_CycleGAN(UNIT):
    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)

        out_A = self.generator(out, self.is_training, scope="generator_A")
        out_B = self.generator(out, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb #, shared

    def generate_a2b(self, x_A):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_B")

        return out #, shared

    def generate_b2a(self, x_B):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_A")

        return out #, shared

    def _build_model(self):
        self._define_input()
        # self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb = self.translation(domain_A, domain_B)
            x_bab = self.generate_a2b(x_ba)
            x_aba = self.generate_b2a(x_ab)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B = self.generate_a2b(domain_A) # for test
            self.fake_A = self.generate_b2a(domain_B) # for test

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss = ops_UNIT.KL_divergence(shared)
        # enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        # enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        # l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_MultiEncSpecificBranchFromImg_CycleGAN_ChangeRes_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb #, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out #, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out #, shared

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss = ops_UNIT.KL_divergence(shared)
        # enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        # enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        # l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 5e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e2*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


# ## Combined with pretrained seg model
# from FCN import fcn8_vgg
# class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_FCN8Loss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
#     def init_net(self, args):
#         assert args.pretrained_path is not None
#         if args.pretrained_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A') \
#                 + tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_B')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverPart = tf.train.Saver(var, max_to_keep=5)
#         if args.pretrained_fcn_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg16_fcn8')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverFcnPart = tf.train.Saver(var, max_to_keep=5)
#         if args.pretrained_vgg_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
#         if args.test_model_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverTest = tf.train.Saver(var, max_to_keep=5)
#         self.summary_writer = tf.summary.FileWriter(args.model_dir)
#         sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
#                 summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
#         if args.phase == 'train':
#             gpu_options = tf.GPUOptions(allow_growth=True)
#         else:
#             gpu_options = tf.GPUOptions(allow_growth=True)
#         sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#         self.sess = sv.prepare_or_wait_for_session(config=sess_config)

#     # ##############################################################################
#     # # BEGIN of ENCODERS
#     # def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
#     #     channel = self.ngf
#     #     with tf.variable_scope(scope, reuse=reuse) :
#     #         x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

#     #         for i in range(1, self.n_encoder) :
#     #             x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
#     #             channel *= 2

#     #         # channel = 256
#     #         for i in range(0, self.n_enc_resblock) :
#     #             x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
#     #                          normal_weight_init=self.normal_weight_init,
#     #                          is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

#     #         return x
#     # # END of ENCODERS
#     # ##############################################################################

#     def _build_model(self):
#         self._define_input()
#         domain_A = self.domain_A
#         domain_B = self.domain_B

#         #### image translation ####
#         feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
#         feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
#         self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
#         self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
#         with tf.variable_scope('UNIT'):
#             """ Define Encoder, Generator, Discriminator """
#             """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
#             x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
#             x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
#             x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
#             self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

#             # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
#             real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
#             real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

#             if self.replay_memory :
#                 self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
#                 self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
#                 # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
#                 fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
#                 fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
#             else :
#                 # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
#                 fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
#                 fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

#             """ Generated Image """
#             self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
#             self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
#             self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
#             self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
#             self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
#             self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

#         #### Extract segmentation ####
#         self.seg_model = fcn8_vgg.FCN8VGG(vgg16_npy_path='./weights/vgg16.npy')
#         input_data = tf.concat([domain_A, domain_B, x_ab, x_ba], axis=0)
#         with tf.variable_scope('vgg16_fcn8') as vs:
#             self.seg_model.build(input_data, train=True, num_classes=self.segment_class, random_init_fc8=True, debug=False, use_dilated=False)
#         self.pred_mask_prob_A, self.pred_mask_prob_B, self.pred_mask_prob_ab, self.pred_mask_prob_ba = tf.split(self.seg_model.upscore32, 4, axis=0)
#         self.pred_mask_A = tf.argmax(self.pred_mask_prob_A[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_B = tf.argmax(self.pred_mask_prob_B[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_ab = tf.argmax(self.pred_mask_prob_ab[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_ba = tf.argmax(self.pred_mask_prob_ba[:,:,:,0:self.segment_class-1], -1)
#         if 8==self.segment_class:
#             self.pred_mask_A += 1
#             self.pred_mask_B += 1
#             self.pred_mask_ab += 1
#             self.pred_mask_ba += 1

#         """ Define Loss """
#         G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
#         G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

#         D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
#         D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

#         enc_loss = ops_UNIT.KL_divergence(shared)
#         enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
#         enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

#         l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
#         l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
#         l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
#         l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

#         content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

#         seg_consis_loss_a = tf.reduce_mean(ops_UNIT.L2_loss(self.pred_mask_prob_A, self.pred_mask_prob_ab)) # semantic consistency
#         seg_consis_loss_b = tf.reduce_mean(ops_UNIT.L2_loss(self.pred_mask_prob_B, self.pred_mask_prob_ba)) # semantic consistency
#         # seg_consis_loss = seg_consis_loss_a + seg_consis_loss_b
#         seg_consis_loss = seg_consis_loss_a

#         Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
#                            self.L1_weight * l1_loss_a + \
#                            self.L1_cycle_weight * l1_loss_aba + \
#                            self.KL_weight * enc_loss + \
#                            self.KL_cycle_weight * enc_bab_loss

#         Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
#                            self.L1_weight * l1_loss_b + \
#                            self.L1_cycle_weight * l1_loss_bab + \
#                            self.KL_weight * enc_loss + \
#                            self.KL_cycle_weight * enc_aba_loss

#         Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
#         Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

#         self.Generator_loss = Generator_A_loss + Generator_B_loss + 0.*style_loss + 0.*content_loss + 1.*seg_consis_loss
#         self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
#         self.D_loss_A = Discriminator_A_loss
#         self.D_loss_B = Discriminator_B_loss


#         """ Training """
#         t_vars = tf.trainable_variables()
#         G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
#         D_vars = [var for var in t_vars if 'discriminator' in var.name]


#         # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#         # pdb.set_trace()
#         self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
#         self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

#         """" Summary """
#         self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
#         self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
#         self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
#         self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
#         self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
#         self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
#         self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
#         self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
#         self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
#         self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
#         self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)
#         self.seg_consis_loss_a_sum = tf.summary.scalar("seg_consis_loss_a", seg_consis_loss_a)
#         self.seg_consis_loss_b_sum = tf.summary.scalar("seg_consis_loss_b", seg_consis_loss_b)

#         self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
#         self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
#         self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
#         self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
#         self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
#         self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

#         self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
#                         self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
#                         self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
#                         self.enc_bab_loss_sum, self.enc_aba_loss_sum, 
#                         self.content_loss_sum, self.style_loss_sum, self.seg_consis_loss_a_sum, self.seg_consis_loss_b_sum])
#         self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

#         self.init_op = tf.global_variables_initializer()
#         self._define_output()

#     def sample_model(self, sample_dir, epoch, idx):
#         img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, feaMapA, feaMapB, \
#             pred_mask_A, pred_mask_B, pred_mask_ab, pred_mask_ba = self.sess.run(
#                 [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
#                 self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
#                 self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.feaMask_list_A[0], self.feaMask_list_B[0], \
#                 self.pred_mask_A, self.pred_mask_B, self.pred_mask_ab, self.pred_mask_ba], \
#                 feed_dict={self.is_training : False}
#             )

#         real_A_img = unprocess_image(real_A, 127.5, 127.5)
#         real_B_img = unprocess_image(real_B, 127.5, 127.5)
#         fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
#         fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
#         fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
#         fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
#         fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
#         fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
#         x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
#         x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
#         x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
#         x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
#         img_name_A = img_name_A[0]
#         img_name_B = img_name_B[0]
#         save_images(real_A_img, [self.batch_size, 1],
#                     '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(real_B_img, [self.batch_size, 1],
#                     '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_img, [self.batch_size, 1],
#                     '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_img, [self.batch_size, 1],
#                     '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_feaMasked_img, [self.batch_size, 1],
#                     '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_feaMasked_img, [self.batch_size, 1],
#                     '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_insNormed_img, [self.batch_size, 1],
#                     '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_insNormed_img, [self.batch_size, 1],
#                     '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(x_aa_img, [self.batch_size, 1],
#                     '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(x_ba_img, [self.batch_size, 1],
#                     '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(x_ab_img, [self.batch_size, 1],
#                     '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(x_bb_img, [self.batch_size, 1],
#                     '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

#         for i in range(self.batch_size):
#             # pdb.set_trace()
#             fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
#                 [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
#                 feed_dict={self.is_training : False, 
#                             self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
#                             self.real_B : real_B}
#             )
#             fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
#             fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
#             x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
#             x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
#             save_images(fake_A_tmp_img, [self.batch_size, 1],
#                         '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(fake_B_tmp_img, [self.batch_size, 1],
#                         '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(x_ba_tmp_img, [self.batch_size, 1],
#                         '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(x_ab_tmp_img, [self.batch_size, 1],
#                         '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            
#         def vis_proc_func(mask):
#             # catId2color = { label.categoryId: label.color for label in labels }
#             trainId2color = { label.trainId: label.color for label in labels }
#             trainId2color[19] = trainId2color[255]
#             ## TODO: seg logits --> one_hot --> catId --> color
#             # seg_catId = seg.argmax(axis=-1)
#             batch_size, img_h, img_w = mask.shape
#             seg_rgb = np.zeros([batch_size, img_h, img_w, 3])
#             for b in range(batch_size):
#                 for h in range(img_h):
#                     for w in range(img_w):
#                         # seg_rgb[b,h,w,:] = catId2color[mask[b,h,w]]
#                         seg_rgb[b,h,w,:] = trainId2color[mask[b,h,w]]
#             return seg_rgb
#         imsave(vis_proc_func(pred_mask_A), [self.batch_size, 1],
#                     '{}/Seg_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_B), [self.batch_size, 1],
#                     '{}/Seg_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_ab), [self.batch_size, 1],
#                     '{}/Seg_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_ba), [self.batch_size, 1],
#                     '{}/Seg_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

#     def train(self, args):
#         """Train SG-GAN"""
#         self.sess.run(self.init_op)
#         self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
#         if args.pretrained_path is not None:
#             self.saverPart.restore(self.sess, args.pretrained_path)
#             print('restored from pretrained_path:', args.pretrained_path)
#         if args.pretrained_fcn_path is not None:
#             self.saverFcnPart.restore(self.sess, args.pretrained_fcn_path)
#             print('restored from pretrained_fcn_path:', args.pretrained_fcn_path)
#         if args.pretrained_vgg_path is not None:
#             self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
#             print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

#         counter = args.global_step
#         start_time = time.time()

#         if 0==args.global_step:
#             if args.continue_train and self.load_last_ckpt(args.model_dir):
#                 print(" [*] Load SUCCESS")
#             else:
#                 print(" [!] Load failed...")
#         else:
#             ## global_step is set manually
#             if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
#                 print(" [*] Load SUCCESS")
#             else:
#                 print(" [!] Load failed...")

#         for epoch in range(args.epoch):
#             # batch_idxs = self.sample_num // self.batch_size
#             batch_idxs = self.sample_num
#             # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
#             # idx = int(epoch/len(self.weights_schedule))
#             # loss_weights = self.weights_schedule[idx]

#             d_loss = np.inf
#             for idx in range(0, batch_idxs):
#                 step_ph = epoch*batch_idxs + idx
#                 num_steps = args.epoch*batch_idxs
#                 lr = args.lr*((1 - counter / num_steps)**0.9)
#                 # Update D
#                 # if d_loss>3.0:
#                 if self.D_optim is not None:
#                     _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
#                                                             feed_dict={self.is_training : True, self.lr : lr})
#                     self.writer.add_summary(summary_str, counter)

#                 # Update G
#                 fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], \
#                                                                         feed_dict={self.is_training : True, self.lr : lr})
#                 self.writer.add_summary(summary_str, counter)

#                 counter += 1
#                 if np.mod(counter, args.print_freq) == 1:
#                     # display training status
#                     print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
#                           % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
#                     self.sample_model(args.sample_dir, epoch, counter)

#                 if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
#                     self.save(args.model_dir, counter)