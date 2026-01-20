
import os
import re
from collections.abc import Iterable
from collections.abc import Mapping as ABCMapping
from itertools import chain, repeat
from typing import FrozenSet, List, Mapping, Tuple

import attr
from cytoolz import dissoc, first, keyfilter, memoize, merge, merge_with, second
from frozendict import frozendict
from exprmat.data.io import zopen
from exprmat.ansi import error, warning, info


def convert(genes):
    # genes supplied as dictionary.
    if isinstance(genes, ABCMapping):
        return frozendict(genes)
    # genes supplied as iterable of (gene, weight) tuples.
    elif isinstance(genes, Iterable) and all(isinstance(n, tuple) for n in genes):
        return frozendict(genes)
    # genes supplied as iterable of genes.
    elif isinstance(genes, Iterable) and all(isinstance(n, str) for n in genes):
        return frozendict(zip(genes, repeat(1.0)))


@attr.s()
class signature():
    """
    Gene signatures, i.e. a set of genes that are biologically related.
    """

    @classmethod
    def from_gmt(
        cls, fname: str, field_separator: str = "\t", gene_separator: str = "\t"
    ):
        assert os.path.exists(fname), f'"{fname}" does not exist.'
        def signatures():
            with zopen(fname, "r") as file:
                for line in file:
                    if isinstance(line, (bytes, bytearray)):
                        line = line.decode()
                    if line.startswith("#") or not line.strip():
                        continue
                    name, _, genes_str = re.split(
                        field_separator, line.rstrip(), maxsplit=2
                    )
                    genes = genes_str.split(gene_separator)
                    yield signature(name = name, gene2weight = genes)

        return list(signatures())


    @classmethod
    def to_gmt(
        cls,
        fname: str,
        signatures: List["signature"],
        field_separator: str = "\t",
        gene_separator: str = "\t",
    ):
        with zopen(fname, "wt") as file:
            for signature in signatures:
                genes = gene_separator.join(signature.genes)
                file.write(
                    f"{signature.name},{field_separator},{signature.metadata(',')},"
                    f"{field_separator}{genes}\n"
                )


    @classmethod
    def from_grp(cls, fname: str, name: str):
        
        assert os.path.exists(fname), f'"{fname}" does not exist.'
        with zopen(fname, "r") as file:
            return signature(
                name = name,
                gene2weight = [
                    line.rstrip()
                    for line in file
                    if not line.startswith("#") and line.strip()
                ],
            )

    @classmethod
    def from_rnk(cls, fname: str, name: str, field_separator=",") -> "signature":
        """
        Reads in a signature from an RNK file. This format associates weights with 
        the genes part of the signature.
        """
        
        assert os.path.exists(fname), f'"{fname}" does not exist.'
        def columns():
            with zopen(fname, "r") as file:
                for line in file:
                    if line.startswith("#") or not line.strip():
                        continue
                    columns = tuple(map(str.rstrip, re.split(field_separator, line)))
                    assert len(columns) == 2, "Invalid file format."
                    yield columns

        return signature(name = name, gene2weight = list(columns()))


    name: str = attr.ib()
    gene2weight: Mapping[str, float] = attr.ib(converter=convert)


    @name.validator
    def name_validator(self, attribute, value) -> None:
        if len(value) == 0:
            error("a gene signature must have a non-empty name.")


    @gene2weight.validator
    def gene2weight_validator(self, attribute, value) -> None:
        if len(value) == 0:
            error("a gene signature must have at least one gene.")


    @property
    @memoize
    def genes(self) -> Tuple[str, ...]:
        """
        Return genes in this signature. 
        Genes are sorted in descending order according to weight.
        """
        return tuple(
            map(first, sorted(self.gene2weight.items(), key = second, reverse = True))
        )


    @property
    @memoize
    def weights(self) -> Tuple[float, ...]:
        """
        Return the weights of the genes in this signature. 
        Genes are sorted in descending order according to weight.
        """
        return tuple(
            map(second, sorted(self.gene2weight.items(), key = second, reverse = True))
        )


    def metadata(self):
        """
        Textual representation of metadata for this signature.
        Used as description when storing this signature as part of a GMT file.
        """
        return ""


    def copy(self, **kwargs):
        try: return signature(**merge(vars(self), kwargs))
        except TypeError: error('invalid or obsolete objects cannot be loaded.')


    def rename(self, name: str):
        return self.copy(name = name)


    def add(self, gene_symbol: str, weight: float = 1.0):
        return self.copy(
            gene2weight = list(chain(self.gene2weight.items(), [(gene_symbol, weight)]))
        )


    def union(self, other: "signature") -> "signature":
        """
        Creates a new instance which is the union of this signature and the other supplied
        signature. The weight associated with the genes in the intersection is the maximum 
        of the weights in the composing signatures.
        """

        return self.copy(
            name = f"({self.name} | {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight = frozendict(
                merge_with(max, self.gene2weight, other.gene2weight))
        )


    def difference(self, other):
        """
        Creates a new instance which is the difference of this signature and the supplied other
        signature. The weight associated with the genes in the difference are taken from 
        this gene signature.
        """

        return self.copy(
            name = f"({self.name} - {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight = frozendict(dissoc(dict(self.gene2weight), *other.genes)),
        )


    def intersection(self, other):
        """
        Creates a new instance which is the intersection of this signature and the supplied 
        other signature. The weight associated with the genes in the intersection is the 
        maximum of the weights in the composing signatures.
        """

        genes = set(self.gene2weight.keys()).intersection(set(other.gene2weight.keys()))
        return self.copy(
            name = f"({self.name} & {other.name})"
            if self.name != other.name
            else self.name,
            gene2weight = frozendict(
                keyfilter(
                    lambda k: k in genes,
                    merge_with(max, self.gene2weight, other.gene2weight),
                )
            ),
        )


    def noweights(self):
        """
        Create a new gene signature with uniform weights, 
        i.e. all weights are equal and set to 1.0.
        """
        return self.copy(gene2weight = self.genes)


    def head(self, n: int = 5) -> "signature":
        assert n >= 1, "n must be greater than or equal to one."
        genes = self.genes[0 : n]  # genes are sorted in ascending order according to weight.
        return self.copy(gene2weight = keyfilter(lambda k: k in genes, self.gene2weight))


    def jaccard_index(self, other: "signature") -> float:
        """
        Calculate the symmetrical similarity metric between this and another signature.
        The JI is a value between 0.0 and 1.0.
        """
        ss = set(self.genes)
        so = set(other.genes)
        return float(len(ss.intersection(so))) / len(ss.union(so))


    def __len__(self): return len(self.genes)
    def __contains__(self, item: str) -> bool: return item in self.gene2weight.keys()
    def __getitem__(self, item: str) -> float: return self.gene2weight[item]
    def __str__(self) -> str: return f"[{','.join(self.genes)}]"


@attr.s(frozen = True)
class regulon(signature):
    """
    A regulon is a gene signature that defines the target genes of a 
    transcription factor and thereby defines a subnetwork of a larger gene 
    regulatory network connecting a TF with its target genes.
    """

    gene2occurrence: Mapping[str, float] = attr.ib(converter = convert)
    transcription_factor: str = attr.ib()
    context: FrozenSet[str] = attr.ib(default = frozenset())
    score: float = attr.ib(default = 0.0)
    nes: float = attr.ib(default = 0.0)
    orthologous_identity: float = attr.ib(default = 0.0)
    similarity_qvalue: float = attr.ib(default = 0.0)
    annotation: str = attr.ib(default = "")


    @transcription_factor.validator
    def non_empty(self, attribute, value):
        if len(value) == 0: error("a regulon must have a transcription factor.")


    def metadata(self, field_separator: str = ",") -> str:
        return f"{self.transcription_factor} (score: {self.score})"


    def copy(self, **kwargs) -> "regulon":
        try: return regulon(**merge(vars(self), kwargs))
        except TypeError: error('failed to load legacy format dumps.')


    def union(self, other: signature) -> "regulon":
        assert self.transcription_factor == getattr(
            other, "transcription_factor", self.transcription_factor
        ), "union of two regulons is only possible when same factor."
        
        return (
            super()
            .union(other)
            .copy(
                context = self.context.union(getattr(other, "context", frozenset())),
                score = max(self.score, getattr(other, "score", 0.0)),
            )
        )


    def difference(self, other: signature) -> "regulon":
        assert self.transcription_factor == getattr(
            other, "transcription_factor", self.transcription_factor
        ), "difference of two regulons is only possible when same factor."
        
        return (
            super()
            .difference(other)
            .copy(
                context = self.context.union(getattr(other, "context", frozenset())),
                score = max(self.score, getattr(other, "score", 0.0)),
            )
        )


    def intersection(self, other: signature) -> "regulon":
        assert self.transcription_factor == getattr(
            other, "transcription_factor", self.transcription_factor
        ), "intersection of two regulons is only possible when same factor."
        
        return (
            super()
            .intersection(other)
            .copy(
                context = self.context.union(getattr(other, "context", frozenset())),
                score = max(self.score, getattr(other, "score", 0.0)),
            )
        )
