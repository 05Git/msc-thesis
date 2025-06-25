from abc import ABC

class PolicyDistillationAlgorithm(ABC):
    def set_teacher(self, teacher_model, teacher_env=None):
        """
        Specify or replace teacher model to use for policy distillation.
        ProximalPolicyDistillation will create a separate policy for the student.

        :param teacher_model: SB3 [On/Off]PolicyAlgorithm object to use as teacher for distillation.
        :param teacher_env: Gymnasium Env wrapped in VecNormalize, with teacher normalization statistics.
                            This is required by Mujoco and environments that require normalizing observations.
        """

        self.teacher_model = teacher_model
        self.teacher_env = teacher_env

    def _excluded_save_params(self):
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.
        :return: List of parameters that should be excluded from being saved with pickle.
        """
        excluded = super()._excluded_save_params()
        return excluded + ['teacher_model']
