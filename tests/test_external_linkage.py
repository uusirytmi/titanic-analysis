import unittest
import pandas as pd
from experiments.external_linkage import normalize,link_identity,attach_labels,validate_candidate


def source():
    return pd.DataFrame(dict(Name=['Person, Mr. One','Person, Mrs. Two'],Ticket=['A/1','B 2'],Pclass=[1,2],Sex=['male','female'],Survived=[0,1],boat=[None,'3'],body=[5,None]))

def target():
    return source().drop(columns=['Survived','boat','body']).assign(PassengerId=[892,893])

class RecordLinkage(unittest.TestCase):
    def test_normalization_and_exact_composite(self):
        self.assertEqual(normalize('É. X-1'), 'ex1')
        self.assertEqual(normalize(None),'')
        t=target();t.loc[0,'Name']='PERSON MR ONE';t.loc[0,'Ticket']='a1'
        m=link_identity(t,source());self.assertTrue((m.status=='matched').all())
        self.assertEqual(attach_labels(m,source()).Survived.tolist(),[0,1])
    def test_no_name_only_match_or_missing_key_fallback(self):
        t=target();t.loc[0,'Ticket']='different';t.loc[1,'Sex']=None
        m=link_identity(t,source())
        self.assertEqual(m.status.tolist(),['unmatched','missing_key'])
        self.assertTrue(attach_labels(m,source()).Survived.isna().all())
    def test_duplicate_source_never_resolved_even_if_labels_agree(self):
        s=pd.concat([source(),source().iloc[[0]]],ignore_index=True)
        m=link_identity(target(),s)
        self.assertEqual(m.status.iloc[0],'ambiguous_source_key')
        self.assertEqual(m.candidate_count.iloc[0],2)
        self.assertTrue(pd.isna(attach_labels(m,s).Survived.iloc[0]))
        s.loc[2,'Survived']=1
        pd.testing.assert_frame_equal(m,link_identity(target(),s))
    def test_duplicate_target_not_assigned_twice(self):
        t=pd.concat([target(),target().iloc[[0]].assign(PassengerId=894)],ignore_index=True)
        m=link_identity(t,source())
        self.assertEqual((m.status=='duplicate_target_key').sum(),2)
    def test_labels_and_post_outcome_fields_do_not_select_identity(self):
        t=target();s=source();a=link_identity(t,s)
        t['Survived']=[1,0];s['Survived']=[1,0];s['boat']='changed';s['body']=999
        pd.testing.assert_frame_equal(a,link_identity(t,s))
        pd.testing.assert_frame_equal(a,link_identity(t.iloc[::-1],s.iloc[::-1]))
    def test_submission_validation(self):
        ids=list(range(892,1310));candidate=pd.DataFrame(dict(PassengerId=ids,Survived=[0]*418))
        validate_candidate(candidate,ids)
        for bad in [candidate.iloc[::-1],candidate.iloc[:-1],candidate.assign(Survived=2),candidate.assign(PassengerId=892)]:
            with self.assertRaises(AssertionError):validate_candidate(bad,ids)

if __name__=='__main__':unittest.main()
