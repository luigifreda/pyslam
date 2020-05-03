from .network import Network


class ResNet50(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 64, 2, padding=3, relu=False, name='conv1')
             .batch_normalization(relu=True, scale=True, center=True, name='bn_conv1')
             .max_pool(3, 2, name='pool1')
             .conv(1, 256, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(scale=True, center=True, name='bn2a_branch1'))

        (self.feed('pool1')
             .conv(1, 64, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(scale=True, center=True, relu=True, name='bn2a_branch2a')
             .conv(3, 64, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(scale=True, center=True, relu=True, name='bn2a_branch2b')
             .conv(1, 256, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(scale=True, center=True, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1',
                   'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu')
         .conv(1, 64, 1, biased=False, relu=False, name='res2b_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn2b_branch2a')
         .conv(3, 64, 1, biased=False, relu=False, name='res2b_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn2b_branch2b')
         .conv(1, 256, 1, biased=False, relu=False, name='res2b_branch2c')
         .batch_normalization(scale=True, center=True, name='bn2b_branch2c'))

        (self.feed('res2a_relu',
                   'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu')
         .conv(1, 64, 1, biased=False, relu=False, name='res2c_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn2c_branch2a')
         .conv(3, 64, 1, biased=False, relu=False, name='res2c_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn2c_branch2b')
         .conv(1, 256, 1, biased=False, relu=False, name='res2c_branch2c')
         .batch_normalization(scale=True, center=True, name='bn2c_branch2c'))

        (self.feed('res2b_relu',
                   'bn2c_branch2c')
         .add(name='res2c')
         .relu(name='res2c_relu')
         .conv(1, 512, 2, biased=False, relu=False, name='res3a_branch1')
         .batch_normalization(scale=True, center=True, name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 128, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(scale=True, center=True, relu=True, name='bn3a_branch2a')
             .conv(3, 128, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(scale=True, center=True, relu=True, name='bn3a_branch2b')
             .conv(1, 512, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(scale=True, center=True, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1',
                   'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu')
         .conv(1, 128, 1, biased=False, relu=False, name='res3b_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3b_branch2a')
         .conv(3, 128, 1, biased=False, relu=False, name='res3b_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3b_branch2b')
         .conv(1, 512, 1, biased=False, relu=False, name='res3b_branch2c')
         .batch_normalization(scale=True, center=True, name='bn3b_branch2c'))

        (self.feed('res3a_relu',
                   'bn3b_branch2c')
         .add(name='res3b')
         .relu(name='res3b_relu')
         .conv(1, 128, 1, biased=False, relu=False, name='res3c_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3c_branch2a')
         .conv(3, 128, 1, biased=False, relu=False, name='res3c_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3c_branch2b')
         .conv(1, 512, 1, biased=False, relu=False, name='res3c_branch2c')
         .batch_normalization(scale=True, center=True, name='bn3c_branch2c'))

        (self.feed('res3b_relu',
                   'bn3c_branch2c')
         .add(name='res3c')
         .relu(name='res3c_relu')
         .conv(1, 128, 1, biased=False, relu=False, name='res3d_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3d_branch2a')
         .conv(3, 128, 1, biased=False, relu=False, name='res3d_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn3d_branch2b')
         .conv(1, 512, 1, biased=False, relu=False, name='res3d_branch2c')
         .batch_normalization(scale=True, center=True, name='bn3d_branch2c'))

        (self.feed('res3c_relu',
                   'bn3d_branch2c')
         .add(name='res3d')
         .relu(name='res3d_relu')
         .conv(1, 1024, 2, biased=False, relu=False, name='res4a_branch1')
         .batch_normalization(scale=True, center=True, name='bn4a_branch1'))

        (self.feed('res3d_relu')
             .conv(1, 256, 2, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(scale=True, center=True, relu=True, name='bn4a_branch2a')
             .conv(3, 256, 1, biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(scale=True, center=True, relu=True, name='bn4a_branch2b')
             .conv(1, 1024, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(scale=True, center=True, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1',
                   'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu')
         .conv(1, 256, 1, biased=False, relu=False, name='res4b_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4b_branch2a')
         .conv(3, 256, 1, biased=False, relu=False, name='res4b_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4b_branch2b')
         .conv(1, 1024, 1, biased=False, relu=False, name='res4b_branch2c')
         .batch_normalization(scale=True, center=True, name='bn4b_branch2c'))

        (self.feed('res4a_relu',
                   'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu')
         .conv(1, 256, 1, biased=False, relu=False, name='res4c_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4c_branch2a')
         .conv(3, 256, 1, biased=False, relu=False, name='res4c_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4c_branch2b')
         .conv(1, 1024, 1, biased=False, relu=False, name='res4c_branch2c')
         .batch_normalization(scale=True, center=True, name='bn4c_branch2c'))

        (self.feed('res4b_relu',
                   'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu')
         .conv(1, 256, 1, biased=False, relu=False, name='res4d_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4d_branch2a')
         .conv(3, 256, 1, biased=False, relu=False, name='res4d_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4d_branch2b')
         .conv(1, 1024, 1, biased=False, relu=False, name='res4d_branch2c')
         .batch_normalization(scale=True, center=True, name='bn4d_branch2c'))

        (self.feed('res4c_relu',
                   'bn4d_branch2c')
         .add(name='res4d')
         .relu(name='res4d_relu')
         .conv(1, 256, 1, biased=False, relu=False, name='res4e_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4e_branch2a')
         .conv(3, 256, 1, biased=False, relu=False, name='res4e_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4e_branch2b')
         .conv(1, 1024, 1, biased=False, relu=False, name='res4e_branch2c')
         .batch_normalization(scale=True, center=True, name='bn4e_branch2c'))

        (self.feed('res4d_relu',
                   'bn4e_branch2c')
         .add(name='res4e')
         .relu(name='res4e_relu')
         .conv(1, 256, 1, biased=False, relu=False, name='res4f_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4f_branch2a')
         .conv(3, 256, 1, biased=False, relu=False, name='res4f_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn4f_branch2b')
         .conv(1, 1024, 1, biased=False, relu=False, name='res4f_branch2c')
         .batch_normalization(scale=True, center=True, name='bn4f_branch2c'))

        (self.feed('res4e_relu',
                   'bn4f_branch2c')
         .add(name='res4f')
         .relu(name='res4f_relu')
         .conv(1, 2048, 2, biased=False, relu=False, name='res5a_branch1')
         .batch_normalization(scale=True, center=True, name='bn5a_branch1'))

        (self.feed('res4f_relu')
             .conv(1, 512, 2, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(scale=True, center=True, relu=True, name='bn5a_branch2a')
             .conv(3, 512, 1, biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(scale=True, center=True, relu=True, name='bn5a_branch2b')
             .conv(1, 2048, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(scale=True, center=True, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
         .add(name='res5a')
         .relu(name='res5a_relu')
         .conv(1, 512, 1, biased=False, relu=False, name='res5b_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn5b_branch2a')
         .conv(3, 512, 1, biased=False, relu=False, name='res5b_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn5b_branch2b')
         .conv(1, 2048, 1, biased=False, relu=False, name='res5b_branch2c')
         .batch_normalization(scale=True, center=True, name='bn5b_branch2c'))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
         .add(name='res5b')
         .relu(name='res5b_relu')
         .conv(1, 512, 1, biased=False, relu=False, name='res5c_branch2a')
         .batch_normalization(scale=True, center=True, relu=True, name='bn5c_branch2a')
         .conv(3, 512, 1, biased=False, relu=False, name='res5c_branch2b')
         .batch_normalization(scale=True, center=True, relu=True, name='bn5c_branch2b')
         .conv(1, 2048, 1, biased=False, relu=False, name='res5c_branch2c')
         .batch_normalization(scale=True, center=True, name='bn5c_branch2c'))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
         .add(name='res5c')
         .relu(name='res5c_relu')
         .avg_pool(7, 1, padding='VALID', name='pool5'))

        if not self.fcn:
            (self.feed('pool5')
             .fc(1000, relu=False, name='fc1000')
             .softmax(name='prob'))
