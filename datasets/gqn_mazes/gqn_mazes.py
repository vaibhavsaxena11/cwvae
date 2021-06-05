"""gqn_mazes dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

_DESCRIPTION = """
# GQN Mazes Dataset

References:
```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```
```
@article {Eslami1204,
	title = {Neural scene representation and rendering},
	author = {Eslami, S. M. Ali and Jimenez Rezende, Danilo and Besse, Frederic and Viola, Fabio and Morcos, Ari S. and Garnelo, Marta and Ruderman, Avraham and Rusu, Andrei A. and Danihelka, Ivo and Gregor, Karol and Reichert, David P. and Buesing, Lars and Weber, Theophane and Vinyals, Oriol and Rosenbaum, Dan and Rabinowitz, Neil and King, Helen and Hillier, Chloe and Botvinick, Matt and Wierstra, Daan and Kavukcuoglu, Koray and Hassabis, Demis},
	doi = {10.1126/science.aar6170},
	publisher = {American Association for the Advancement of Science},
	URL = {https://science.sciencemag.org/content/360/6394/1204},
	journal = {Science},
	year = {2018},
}
```
"""

_CITATION = """
@article{saxena2021clockwork,
      title={Clockwork Variational Autoencoders}, 
      author={Vaibhav Saxena and Jimmy Ba and Danijar Hafner},
      year={2021},
      eprint={2102.09532},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_DOWNLOAD_URL = "https://archive.org/download/gqn_mazes/gqn_mazes.zip"


class GqnMazes(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for GQN Mazes dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "video": tfds.features.Video(shape=(None, 64, 64, 3)),
                }
            ),
            supervised_keys=None,
            homepage="https://archive.org/details/gqn_mazes",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(_DOWNLOAD_URL)

        return {
            "train": self._generate_examples(path / "train"),
            "test": self._generate_examples(path / "test"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob("*.mp4"):
            yield str(f), {
                "video": str(f.resolve()),
            }
