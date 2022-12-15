import citeme

@citeme.article({momin2021mathematical,
  title={Mathematical Modeling of Heat Conduction},
  author={Momin, Abdul Aziz and Shende, Nikhil and Anamtatmakula, Abhijna and Ganguly, Emily and Gurbani, Ashwin and Joshi, Chaitanya A and Mahajan, Yogesh Y},
  journal={arXiv preprint arXiv:2107.11737},
  year={2021}
})
def my_func():
    return True

# Calling the function will add the citation to the
# library
my_func()

# Write all the citations to a bibtex file
write_to_bibtex('called.bib')


