/*
 *  LexicalChainModel.Cpp
 *
 *  Copyright 2015 By Laura Mascarell. University of Zurich. All rights reserved.
 *
 *  This file is part of Docent, a document-level decoder for phrase-based
 *  statistical machine translation.
 *
 *  Docent is free software: you can redistribute it and/or modify it under the
 *  terms of the GNU General Public License as published by the Free Software
 *  Foundation, either version 3 of the License, or (at your option) any later
 *  version.
 *
 *  Docent is distributed in the hope that it will be useful, but WITHOUT ANY
 *  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 *  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 *  details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  Docent. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Docent.h"
#include "DocumentState.h"
#include "FeatureFunction.h"
#include "SearchStep.h"
//#include "SemanticSpace.h"
#include "LexicalChainModel.h"
#include <iostream>
#include "MMAXDocument.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>

#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/unordered_map.hpp>

namespace ublas = boost::numeric::ublas;

//using namespace std;

LexicalChainModel::LexicalChainModel(const Parameters &params) : 
  logger_("LexicalChainModel") {  
  sspace_ = SemanticSpace::load(params.get<std::string>("word-model-file"));
}

LexicalChainModel::~LexicalChainModel() {
  delete sspace_;
}

struct LexicalChainModelState : public FeatureFunction::State, public FeatureFunction::StateModifications {
  LexicalChainModelState(uint nsents) {}

  std::map<uint, std::map<uint, std::map<std::string, std::vector<Relation* > > > > ids_;
  std::vector<Relation* > relations_;


  void addRelation(const uint sentno, const uint wordno, const uint relid) {
    if ( relations_.empty() or relid > (relations_.size()-1) ) { 
      Relation* r = new Relation();
      relations_.push_back(r);
      ids_[sentno][wordno]["from"].push_back(r);

    }
    else{
      ids_[sentno][wordno]["to"].push_back(relations_[relid]);
    }
  }


  void addPhrasePair(const uint sentno, const AnchoredPhrasePair& app) {
    WordAlignment wa = app.second.get().getWordAlignment();
    PhraseData sd = app.second.get().getSourcePhrase().get();
    PhraseData td = app.second.get().getTargetPhrase().get();

    uint wordno = app.first.find_first();
    for (uint j=0; j<sd.size(); ++j) {
      if ( ids_.find(sentno) != ids_.end() and ids_[sentno].find(wordno) != ids_[sentno].end() ) {
     
	std::string translation;
        for (WordAlignment::const_iterator wit = wa.begin_for_source(j); wit != wa.end_for_source(j); ++wit)
          translation += td[*wit];
     
	std::vector<Relation* > relFrom = ids_[sentno][wordno]["from"];

	for (uint i = 0; i < relFrom.size(); i = i + 1) {
	  relFrom[i]->setFrom(translation);

	}
	std::vector<Relation* > relTo = ids_[sentno][wordno]["to"];

	for (uint i = 0; i < relTo.size(); i = i + 1) {
	  relTo[i]->setTo(translation);
	}

      }
      wordno += 1;
    }
  };


  void removePhrasePair(const uint sentno, const AnchoredPhrasePair& app) {
    WordAlignment wa = app.second.get().getWordAlignment();
    PhraseData sd = app.second.get().getSourcePhrase().get();
    PhraseData td = app.second.get().getTargetPhrase().get();

    uint wordno = app.first.find_first();

    for (uint j=0; j<sd.size(); ++j) {
      if ( ids_.find(sentno) != ids_.end() and ids_[sentno].find(wordno) != ids_[sentno].end() ) {
	std::vector<Relation* > relFrom = ids_[sentno][wordno]["from"];

	for (uint i = 0; i < relFrom.size(); i = i + 1)
	  relFrom[i]->clearFrom();

	std::vector<Relation* > relTo = ids_[sentno][wordno]["to"];

	for (uint i = 0; i < relTo.size(); i = i + 1)
	  relTo[i]->clearTo();
      }
      wordno += 1;
    }
  };


  Float score(const SemanticSpace *sspace_) const {

    Float score = 0;
 
    for (uint i = 0; i < relations_.size(); i = i + 1) {
      const SemanticSpace::WordVector *ssvec1 = NULL;
      const SemanticSpace::WordVector *ssvec2 = NULL;
  
      ssvec1 = sspace_->lookup(relations_[i]->getFrom());
      ssvec2 = sspace_->lookup(relations_[i]->getTo());

      Float sim = 0;
      if (ssvec1 != NULL and ssvec2 != NULL)
	sim = ublas::inner_prod(*ssvec1, *ssvec2) / (ublas::norm_2(*ssvec1) * ublas::norm_2(*ssvec2));
      
      relations_[i]->setScore(sim);
      score += sim;
    }
    
    return Float(score/relations_.size());
    
  }
  

  virtual LexicalChainModelState *clone() const {
	  return new LexicalChainModelState(*this);
  }
};

struct WordPenaltyCounter : public std::unary_function<const AnchoredPhrasePair &,Float> {
	Float operator()(const AnchoredPhrasePair &ppair) const {
		return Float(-ppair.second.get().getTargetPhrase().get().size());
	};
};

FeatureFunction::State *LexicalChainModel::initDocument(const DocumentState &doc, Scores::iterator sbegin) const {
  boost::shared_ptr<const MMAXDocument> mmax = doc.getInputDocument();
  const std::vector<PhraseSegmentation> &segs = doc.getPhraseSegmentations();
  LexicalChainModelState *s = new LexicalChainModelState(segs.size());
  
  const MarkableLevel &chain_rel = mmax->getMarkableLevel("ch_rel");
  BOOST_FOREACH(const Markable &m, chain_rel) {
    uint snt = m.getSentence();
    const CoverageBitmap &cov = m.getCoverage();
    uint wordno = cov.find_first();
    const std::string& relid = m.getAttribute("relid");
    uint x = std::atoi(relid.c_str());
    const std::string &span = m.getAttribute("span");
    s->addRelation(snt, wordno, x);
  }


  for(uint i = 0; i < segs.size(); i++)
    BOOST_FOREACH(const AnchoredPhrasePair &app, segs[i])
      s->addPhrasePair(i,app);


  *sbegin = s->score(sspace_);
  return s;
}

void LexicalChainModel::computeSentenceScores(const DocumentState &doc, uint sentno,
		const FeatureFunction::State *state, Scores::iterator sbegin) const {
	*sbegin = Float(0);
}

FeatureFunction::StateModifications *LexicalChainModel::estimateScoreUpdate(const DocumentState &doc, const SearchStep &step, const State *state,
		Scores::const_iterator psbegin, Scores::iterator sbegin) const {
	const LexicalChainModelState *prevstate = dynamic_cast<const LexicalChainModelState *>(state);
	LexicalChainModelState *s = prevstate->clone();

	const std::vector<SearchStep::Modification> &mods = step.getModifications();

	for(std::vector<SearchStep::Modification>::const_iterator it = mods.begin(); it != mods.end(); ++it) {
	  uint sentno = it->sentno;
	  PhraseSegmentation::const_iterator from_it = it->from_it;
	  PhraseSegmentation::const_iterator to_it = it->to_it;

	  for (PhraseSegmentation::const_iterator pit=from_it; pit != to_it; pit++) {
	    s->removePhrasePair(sentno, *pit);
	  }

	  BOOST_FOREACH(const AnchoredPhrasePair &app, it->proposal) {
	    s->addPhrasePair(sentno, app);
	  }
	}

	*sbegin = s->score(sspace_);
	return s;
}

FeatureFunction::StateModifications *LexicalChainModel::updateScore(const DocumentState &doc, const SearchStep &step, const State *state,
		FeatureFunction::StateModifications *estmods, Scores::const_iterator psbegin, Scores::iterator estbegin) const {
	return estmods;
}

FeatureFunction::State *LexicalChainModel::applyStateModifications(FeatureFunction::State *oldState, FeatureFunction::StateModifications *modif) const {
	LexicalChainModelState *os = dynamic_cast<LexicalChainModelState *>(oldState);
	LexicalChainModelState *ms = dynamic_cast<LexicalChainModelState *>(modif);
	os->ids_.swap(ms->ids_);
	os->relations_.swap(ms->relations_);
	return oldState;
}
